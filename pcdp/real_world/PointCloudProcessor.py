class PointCloudPreprocessor:
    """
    - 기존 단일뷰 PCM(temporal fused) + occlusion prune + variance filter 유지
    - 멀티뷰(Femto + D405)용 process_global / _occlusion_prune_memory_multiview 추가
    - depth_u16를 그대로 Z-buffer로 사용(빠름): uint16 erosion(min-filter)로 Zmin 생성
    - "공격적 삭제"는 free-space carving 기준으로 margin=0이 가장 공격적:
        delete_free_space: z_mem < (z_now - margin)
      (margin>0이면 덜 공격적 / noise 허용)
    """

    # ----------------------------
    # Orbbec intr/dist helper
    # ----------------------------
    @staticmethod
    def _is_orbbec_intrinsics(obj) -> bool:
        return all(hasattr(obj, a) for a in ("fx", "fy", "cx", "cy"))

    @staticmethod
    def _is_orbbec_distortion(obj) -> bool:
        return all(hasattr(obj, a) for a in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"))

    @staticmethod
    def _as_cv_K_from_orbbec_intrinsics(orbbec_intr):
        K = np.array([[float(orbbec_intr.fx), 0.0, float(orbbec_intr.cx)],
                      [0.0, float(orbbec_intr.fy), float(orbbec_intr.cy)],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return K

    @staticmethod
    def _as_cv_dist_from_orbbec_distortion(orbbec_dist):
        # Orbbec(k1..k6, p1, p2) → OpenCV rational 8계수 [k1,k2,p1,p2,k3,k4,k5,k6]
        k1 = float(orbbec_dist.k1); k2 = float(orbbec_dist.k2); k3 = float(orbbec_dist.k3)
        k4 = float(orbbec_dist.k4); k5 = float(orbbec_dist.k5); k6 = float(orbbec_dist.k6)
        p1 = float(orbbec_dist.p1); p2 = float(orbbec_dist.p2)
        return np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)

    @staticmethod
    def convert_orbbec_depth_params(depth_intrinsics, depth_distortion):
        K = PointCloudPreprocessor._as_cv_K_from_orbbec_intrinsics(depth_intrinsics)
        dist = PointCloudPreprocessor._as_cv_dist_from_orbbec_distortion(depth_distortion)
        return K, dist

    @staticmethod
    def _depth_u16_to_m(depth_u16: np.ndarray, z_unit: str) -> np.ndarray:
        """uint16 depth -> meters(float32). 0은 invalid로 유지."""
        d = depth_u16.astype(np.float32, copy=False)
        z_unit = str(z_unit).lower()
        if z_unit == "m":
            return d
        if z_unit == "mm":
            return d * 1e-3
        if z_unit in ("0.1mm", "0_1mm", "1e-4"):
            return d * 1e-4
        raise ValueError(f"Unknown z_unit={z_unit}")

    # ----------------------------
    # ctor
    # ----------------------------
    def __init__(
        self,
        enable_sampling=False,
        target_num_points=1024,
        enable_transform=True,
        extrinsics_matrix=None,
        enable_cropping=True,
        workspace_bounds=None,
        enable_wrist_camera=True,
        # Variance filter
        enable_filter=False,
        variance_kernel_size=5,
        variance_threshold=50,
        # Open3D Outlier filter (legacy)
        nb_points=10,
        sor_std=1.7,
        use_cuda=True,
        verbose=False,
        # PCM temporal
        enable_temporal=False,
        export_mode='off',
        temporal_voxel_size=0.005,
        temporal_decay=0.96,
        temporal_c_min=0.20,
        temporal_prune_every: int = 1,
        stable_export: bool = False,
        # Occlusion prune
        enable_occlusion_prune: bool = True,
        depth_width: Optional[int] = 320,
        depth_height: Optional[int] = 288,
        K_depth: Optional[Sequence[Sequence[float]]] = None,
        dist_depth: Optional[Sequence[float]] = None,
        depth_rectified: bool = True,          # ✅ (기본) 왜곡 보정된 depth로 가정
        erode_k: int = 1,
        z_unit: str = 'm',                     # 'm' / 'mm' / '0.1mm'
        occl_patch_radius: int = 2,
        # miss-prune
        miss_prune_frames: int = 20,
        miss_min_age: int = 2,
        # free-space delete margin (meters)
        free_space_margin_m: float = 0.0,      # ✅ 공격적으로: 0.0  (값↑ => 덜 삭제)
    ):
        # extrinsics (Femto cam_to_base or similar) default
        if extrinsics_matrix is None:
            self.extrinsics_matrix = np.array([
                [0.007131, -0.91491,  0.403594, 0.05116],
                [-0.994138, 0.003833, 0.02656, -0.00918],
                [-0.025717, -0.403641, -0.914552, 0.50821],
                [0., 0., 0., 1.]
            ], dtype=np.float64)
        else:
            self.extrinsics_matrix = np.array(extrinsics_matrix, dtype=np.float64)

        # workspace bounds
        if workspace_bounds is None:
            self.workspace_bounds = [
                [0.132, 0.715],
                [-0.400, 0.350],
                [-0.100, 0.400]
            ]
        else:
            self.workspace_bounds = workspace_bounds

        # K_depth
        if K_depth is None:
            self._K_depth = np.array([
                [252.69204711914062, 0.0, 166.12030029296875],
                [0.0, 252.65277099609375, 176.21173095703125],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
        else:
            if self._is_orbbec_intrinsics(K_depth):
                self._K_depth = self._as_cv_K_from_orbbec_intrinsics(K_depth)
            else:
                self._K_depth = np.array(K_depth, dtype=np.float64)
        assert self._K_depth.shape == (3, 3), f"K_depth must be 3x3, got {self._K_depth.shape}"

        # dist_depth (rectified면 None로 두는게 맞음)
        self.depth_rectified = bool(depth_rectified)
        if self.depth_rectified:
            self._dist_depth = None
        else:
            if dist_depth is None:
                self._dist_depth = np.array(
                    [11.690222, 5.343991, 0.000065, 0.000014, 0.172997, 12.017323, 9.254467, 1.165690],
                    dtype=np.float64
                )
            else:
                if self._is_orbbec_distortion(dist_depth):
                    self._dist_depth = self._as_cv_dist_from_orbbec_distortion(dist_depth)
                else:
                    self._dist_depth = np.asarray(dist_depth, dtype=np.float64)
                    if self._dist_depth.size not in (4, 5, 8):
                        raise ValueError(f"dist_depth must have 4/5/8 coeffs, got {self._dist_depth.size}")

        # basic flags
        self.target_num_points = int(target_num_points)
        self.nb_points = int(nb_points)
        self.sor_std = float(sor_std)
        self.enable_transform = bool(enable_transform)
        self.enable_cropping = bool(enable_cropping)
        self.enable_sampling = bool(enable_sampling)
        self.enable_wrist_camera = bool(enable_wrist_camera)
        self.enable_filter = bool(enable_filter)
        self.variance_kernel_size = int(variance_kernel_size)
        self.variance_threshold = float(variance_threshold)

        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.verbose = bool(verbose)

        # temporal
        self.enable_temporal = bool(enable_temporal)
        self.export_mode = str(export_mode)
        self._temporal_voxel_size = float(temporal_voxel_size)
        self._temporal_decay = float(temporal_decay)
        self._temporal_c_min = float(temporal_c_min)
        self._prune_every = int(max(1, temporal_prune_every))
        self._stable_export = bool(stable_export)

        # occlusion prune params
        self.enable_occlusion_prune = bool(enable_occlusion_prune)
        self._depth_w = int(depth_width)
        self._depth_h = int(depth_height)
        self._erode_k = int(erode_k)
        self._z_unit = str(z_unit)
        self.occl_patch_radius = int(occl_patch_radius)
        self._free_space_margin_m = float(free_space_margin_m)

        self._miss_prune_frames = int(miss_prune_frames)
        self._miss_min_age = int(miss_min_age)

        # base->cam for Femto (invert extrinsics_matrix)
        self._base_to_cam = None
        if self.enable_occlusion_prune or self.enable_filter:
            self._base_to_cam = np.linalg.inv(self.extrinsics_matrix).astype(np.float64)

        # temporal memory buffers
        self._frame_idx = 0
        self._mem_keys = np.empty((0,), dtype=np.dtype((np.void, 12)))
        self._mem_xyz = np.empty((0, 3), dtype=np.float32)
        self._mem_rgb = np.empty((0, 3), dtype=np.float32)
        self._mem_step = np.empty((0,), dtype=np.int32)
        self._mem_miss = np.empty((0,), dtype=np.int16)

        # Femto cached projection
        self._mem_u = np.empty((0,), dtype=np.int32)
        self._mem_v = np.empty((0,), dtype=np.int32)
        self._mem_zcam = np.empty((0,), dtype=np.float32)

        # age threshold from temporal decay
        if 0.0 < self._temporal_decay < 1.0 and 0.0 < self._temporal_c_min < 1.0:
            self._max_age_steps = int(np.floor(np.log(self._temporal_c_min) / np.log(self._temporal_decay)))
        else:
            self._max_age_steps = 0

        # torch init
        self._device = None
        self._K_depth_t = None
        self._dist_depth_t = None
        self._base_to_cam_t = None

        # cache for pinhole(K_d405)
        self._K_pinhole_cache_key = None
        self._K_pinhole_cache_t = None

        if (self.enable_temporal and self.export_mode == 'fused') or self.enable_occlusion_prune or self.enable_filter:
            if not self.use_cuda:
                raise RuntimeError(
                    "GPU-only path enabled (temporal fused / occlusion prune / variance filter). "
                    "Set use_cuda=True and ensure CUDA is available."
                )
            self._maybe_init_torch_camera()

        if self.verbose:
            print("PointCloudPreprocessor initialized")
            print(f"  - temporal: {self.enable_temporal}, export_mode: {self.export_mode}")
            print(f"  - occlusion_prune: {self.enable_occlusion_prune}, free_space_margin_m: {self._free_space_margin_m}")
            print(f"  - depth_rectified: {self.depth_rectified}")
            print(f"  - CUDA: {self.use_cuda}")

    def __call__(self, points):
        return self.process(points)

    # ----------------------------
    # torch init / camera model (Femto)
    # ----------------------------
    def _maybe_init_torch_camera(self):
        if not self.use_cuda:
            return
        dev = torch.device("cuda")
        self._device = dev

        self._K_depth_t = torch.tensor(self._K_depth, dtype=torch.float32, device=dev)

        if self._dist_depth is None:
            dist8 = np.zeros(8, dtype=np.float64)
        else:
            d = self._dist_depth
            if d.size == 4:
                dist8 = np.array([d[0], d[1], d[2], d[3], 0, 0, 0, 0], dtype=np.float64)
            elif d.size == 5:
                dist8 = np.array([d[0], d[1], d[2], d[3], d[4], 0, 0, 0], dtype=np.float64)
            elif d.size == 8:
                dist8 = d.astype(np.float64, copy=False)
            else:
                dist8 = np.zeros(8, dtype=np.float64)
        self._dist_depth_t = torch.tensor(dist8, dtype=torch.float32, device=dev)

        if self._base_to_cam is not None:
            self._base_to_cam_t = torch.tensor(self._base_to_cam.astype(np.float32), device=dev)

    def _opencv_rational_distort_torch(self, xn: torch.Tensor, yn: torch.Tensor):
        k = self._dist_depth_t  # [k1,k2,p1,p2,k3,k4,k5,k6]
        k1, k2, p1, p2, k3, k4, k5, k6 = k
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        x_tan = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        y_tan = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        xd = xn * radial + x_tan
        yd = yn * radial + y_tan
        return xd, yd

    def _project_points_cam_torch(self, xyz_cam_t: torch.Tensor):
        if xyz_cam_t.numel() == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            empty_i64 = np.array([], dtype=np.int64)
            return empty_i32, empty_i32, empty_f64, empty_i64

        z = xyz_cam_t[:, 2]
        valid = z > 0.0
        if torch.count_nonzero(valid) == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            empty_i64 = np.array([], dtype=np.int64)
            return empty_i32, empty_i32, empty_f64, empty_i64

        pts = xyz_cam_t[valid]
        xn = pts[:, 0] / pts[:, 2]
        yn = pts[:, 1] / pts[:, 2]

        # rectified면 왜곡 없음
        if self._dist_depth is None:
            xd, yd = xn, yn
        else:
            xd, yd = self._opencv_rational_distort_torch(xn, yn)

        fx = self._K_depth_t[0, 0]; fy = self._K_depth_t[1, 1]
        cx = self._K_depth_t[0, 2]; cy = self._K_depth_t[1, 2]

        u = torch.round(fx * xd + cx).to(torch.int32)
        v = torch.round(fy * yd + cy).to(torch.int32)

        inb = (u >= 0) & (u < int(self._depth_w)) & (v >= 0) & (v < int(self._depth_h))
        if torch.count_nonzero(inb) == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            empty_i64 = np.array([], dtype=np.int64)
            return empty_i32, empty_i32, empty_f64, empty_i64

        u = u[inb]
        v = v[inb]
        z_out = pts[inb, 2]

        orig_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)[inb]
        return (
            u.detach().cpu().numpy(),
            v.detach().cpu().numpy(),
            z_out.detach().cpu().numpy().astype(np.float64),
            orig_idx.detach().cpu().numpy().astype(np.int64),
        )

    def _project_base_to_cam_torch(self, xyz_base_np: np.ndarray):
        if xyz_base_np.size == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            empty_i64 = np.array([], dtype=np.int64)
            return empty_i32, empty_i32, empty_f64, empty_i64

        self._maybe_init_torch_camera()
        assert self._base_to_cam_t is not None, "Base→Cam not initialized"

        xyz = torch.from_numpy(xyz_base_np.astype(np.float32, copy=False)).to(self._device)
        ones = torch.ones((xyz.shape[0], 1), dtype=torch.float32, device=self._device)
        xyz1 = torch.cat([xyz, ones], dim=1)
        xyz_cam = (xyz1 @ self._base_to_cam_t.T)[:, :3]
        return self._project_points_cam_torch(xyz_cam)

    # ----------------------------
    # temporal memory core
    # ----------------------------
    def reset_temporal(self):
        self._frame_idx = 0
        self._mem_keys = np.empty((0,), dtype=np.dtype((np.void, 12)))
        self._mem_xyz = np.empty((0, 3), dtype=np.float32)
        self._mem_rgb = np.empty((0, 3), dtype=np.float32)
        self._mem_step = np.empty((0,), dtype=np.int32)
        self._mem_miss = np.empty((0,), dtype=np.int16)
        self._mem_u = np.empty((0,), dtype=np.int32)
        self._mem_v = np.empty((0,), dtype=np.int32)
        self._mem_zcam = np.empty((0,), dtype=np.float32)

    def voxel_keys_from_xyz(self, xyz: np.ndarray):
        grid = np.floor(xyz / self._temporal_voxel_size).astype(np.int32, copy=False)
        grid = np.ascontiguousarray(grid)
        keys = grid.view(np.dtype((np.void, 12))).ravel()
        return keys

    def _frame_unique_torch(self, xyz_np: np.ndarray, rgb_np: np.ndarray, last_wins: bool = False):
        if xyz_np.shape[0] == 0:
            empty_keys = np.empty((0,), dtype=np.dtype((np.void, 12)))
            return xyz_np, rgb_np, empty_keys

        device = torch.device("cuda")
        xyz_t = torch.from_numpy(xyz_np.astype(np.float32, copy=False)).to(device)
        grid_t = torch.floor(xyz_t / float(self._temporal_voxel_size)).to(torch.int32)

        uniq, inv = torch.unique(grid_t, dim=0, return_inverse=True)
        idx_all = torch.arange(grid_t.shape[0], device=device, dtype=torch.int64)

        if last_wins:
            idx_sel = torch.zeros(uniq.shape[0], dtype=torch.int64, device=device)
            idx_sel.scatter_reduce_(0, inv, idx_all, reduce='amax', include_self=False)
        else:
            big = torch.iinfo(torch.int64).max
            idx_sel = torch.full((uniq.shape[0],), big, dtype=torch.int64, device=device)
            idx_sel.scatter_reduce_(0, inv, idx_all, reduce='amin', include_self=True)

        idx_np = idx_sel.detach().cpu().numpy().astype(np.int64)
        xyz_unique = xyz_np[idx_np]
        rgb_unique = rgb_np[idx_np]

        grid_sel = uniq.detach().cpu().numpy().astype(np.int32)
        keys_new = np.ascontiguousarray(grid_sel).view(np.dtype((np.void, 12))).ravel()
        return xyz_unique, rgb_unique, keys_new

    def _merge_into_mem(self, xyz_new: np.ndarray, rgb_new: np.ndarray, now_step: int,
                        keys_new: Optional[np.ndarray] = None):
        if xyz_new.shape[0] == 0:
            return
        if keys_new is None:
            keys_new = self.voxel_keys_from_xyz(xyz_new)

        if self._mem_keys.size == 0:
            self._mem_keys = keys_new.copy()
            self._mem_xyz = xyz_new.astype(np.float32, copy=False).copy()
            self._mem_rgb = rgb_new.astype(np.float32, copy=False).copy()
            self._mem_step = np.full((xyz_new.shape[0],), now_step, dtype=np.int32)
            self._mem_miss = np.zeros((xyz_new.shape[0],), dtype=np.int16)
            if self.enable_occlusion_prune:
                u_add, v_add, z_add = self._project_and_pack(self._mem_xyz)
                self._mem_u, self._mem_v, self._mem_zcam = u_add, v_add, z_add
            return

        common, idx_mem, idx_new = np.intersect1d(self._mem_keys, keys_new, return_indices=True)
        if common.size > 0:
            self._mem_xyz[idx_mem] = xyz_new[idx_new]
            self._mem_rgb[idx_mem] = rgb_new[idx_new]
            self._mem_step[idx_mem] = now_step
            self._mem_miss[idx_mem] = 0
            if self.enable_occlusion_prune:
                u_upd, v_upd, z_upd = self._project_and_pack(xyz_new[idx_new])
                self._mem_u[idx_mem] = u_upd
                self._mem_v[idx_mem] = v_upd
                self._mem_zcam[idx_mem] = z_upd

        mask_new_only = np.ones(keys_new.shape[0], dtype=bool)
        if common.size > 0:
            mask_new_only[idx_new] = False

        if mask_new_only.any():
            add_xyz = xyz_new[mask_new_only]
            add_rgb = rgb_new[mask_new_only]
            add_keys = keys_new[mask_new_only]

            self._mem_keys = np.concatenate([self._mem_keys, add_keys], axis=0)
            self._mem_xyz = np.concatenate([self._mem_xyz, add_xyz], axis=0)
            self._mem_rgb = np.concatenate([self._mem_rgb, add_rgb], axis=0)
            self._mem_step = np.concatenate(
                [self._mem_step, np.full((add_xyz.shape[0],), now_step, dtype=np.int32)], axis=0
            )
            self._mem_miss = np.concatenate(
                [self._mem_miss, np.zeros((add_xyz.shape[0],), dtype=np.int16)], axis=0
            )
            if self.enable_occlusion_prune:
                u_add, v_add, z_add = self._project_and_pack(add_xyz)
                self._mem_u = np.concatenate([self._mem_u, u_add], axis=0)
                self._mem_v = np.concatenate([self._mem_v, v_add], axis=0)
                self._mem_zcam = np.concatenate([self._mem_zcam, z_add], axis=0)

    def _prune_mem(self, now_step: int):
        if self._mem_keys.size == 0:
            return
        if (now_step % self._prune_every) != 0:
            return
        age = now_step - self._mem_step
        keep = (age <= self._max_age_steps)
        if not np.all(keep):
            self._delete_mask(keep)

    def _export_array_from_mem(self, now_step: int) -> np.ndarray:
        N = self._mem_keys.size
        if N == 0:
            return np.zeros((0, 7), dtype=np.float32)
        age = (now_step - self._mem_step).astype(np.float32)
        c = (self._temporal_decay ** age).astype(np.float32, copy=False)

        if self._stable_export:
            order = np.argsort(self._mem_keys)
            xyz = self._mem_xyz[order]
            rgb = self._mem_rgb[order]
            c = c[order]
        else:
            xyz = self._mem_xyz
            rgb = self._mem_rgb

        return np.concatenate([xyz, rgb, c[:, None]], axis=1).astype(np.float32, copy=False)

    # ----------------------------
    # rasterize / min filter (GPU)
    # ----------------------------
    def _rasterize_min_float_torch(self, u: np.ndarray, v: np.ndarray, z_m: np.ndarray, H: int, W: int):
        assert self.use_cuda
        if u.size == 0:
            return np.zeros((H, W), dtype=np.float32)

        device = self._device
        pix = torch.from_numpy((v * W + u).astype(np.int64)).to(device, non_blocking=True)
        zt = torch.from_numpy(z_m.astype(np.float32, copy=False)).to(device, non_blocking=True)

        Z = torch.full((H * W,), float("inf"), device=device, dtype=torch.float32)
        try:
            Z = torch.scatter_reduce(Z, 0, pix, zt, reduce='amin', include_self=True)
        except TypeError:
            Z.scatter_reduce_(0, pix, zt, reduce='amin', include_self=True)

        Z = Z.view(H, W)
        Z[torch.isinf(Z)] = 0.0
        return Z.detach().cpu().numpy()

    def _min_filter2d_float_torch(self, Z: np.ndarray, k: int):
        assert self.use_cuda
        if k <= 1:
            return Z
        import torch.nn.functional as F

        device = self._device
        Zt = torch.from_numpy(Z.astype(np.float32, copy=False)).to(device, non_blocking=True)
        sent = torch.tensor(1e6, device=device, dtype=torch.float32)
        Zt = torch.where(Zt <= 0.0, sent, Zt)

        pad = k // 2
        Zn = (-Zt).view(1, 1, *Zt.shape)
        Zn = F.pad(Zn, (pad, pad, pad, pad), mode='replicate')
        Zmax = F.max_pool2d(Zn, kernel_size=k, stride=1, padding=0)
        Zmin = (-Zmax).squeeze(0).squeeze(0)

        Zmin = torch.where(Zmin >= sent * 0.999, torch.tensor(0.0, device=device), Zmin)
        return Zmin.detach().cpu().numpy()

    # ----------------------------
    # depth-u16 min filter (CPU, fast) : ignore 0
    # ----------------------------
    def _min_filter_u16_ignore0(self, depth_u16: np.ndarray, k: int) -> np.ndarray:
        """
        uint16 depth에 대해 min-filter(erosion) 수행.
        0은 invalid이므로 sentinel(65535)로 바꿨다가 복원.
        """
        if k <= 1:
            return depth_u16
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        tmp = depth_u16.copy()
        zero = (tmp == 0)
        tmp[zero] = 65535
        tmp = cv2.erode(tmp, kernel)
        tmp[zero] = 0
        return tmp

    def _zmin_from_depth_u16(self, depth_u16: np.ndarray, *, z_unit: str, patch_radius: int) -> Optional[np.ndarray]:
        """
        depth_u16 -> Zmin(meters, float32)
        - (옵션) erode_k: 센서 노이즈/홀 메움용 min
        - (필수) patch_radius: occlusion_patch용 Zmin
        """
        if depth_u16 is None:
            return None
        if depth_u16.ndim != 2:
            raise ValueError(f"depth_u16 must be HxW, got {depth_u16.shape}")

        d = depth_u16
        if self._erode_k > 0:
            k0 = 2 * self._erode_k + 1
            d = self._min_filter_u16_ignore0(d, k0)

        if patch_radius > 0:
            k1 = 2 * int(patch_radius) + 1
            d = self._min_filter_u16_ignore0(d, k1)

        Z = self._depth_u16_to_m(d, z_unit=z_unit)  # meters float32
        return Z

    # ----------------------------
    # pack / delete
    # ----------------------------
    def _project_and_pack(self, xyz_base: np.ndarray):
        assert self.use_cuda
        n = xyz_base.shape[0]
        u_arr = np.full((n,), -1, dtype=np.int32)
        v_arr = np.full((n,), -1, dtype=np.int32)
        z_arr = np.zeros((n,), dtype=np.float32)

        u, v, z, in_idx = self._project_base_to_cam_torch(xyz_base)
        if in_idx.size > 0:
            u_arr[in_idx] = u
            v_arr[in_idx] = v
            z_arr[in_idx] = z.astype(np.float32, copy=False)
        return u_arr, v_arr, z_arr

    def _delete_mask(self, keep_mask: np.ndarray):
        self._mem_keys = self._mem_keys[keep_mask]
        self._mem_xyz = self._mem_xyz[keep_mask]
        self._mem_rgb = self._mem_rgb[keep_mask]
        self._mem_step = self._mem_step[keep_mask]
        self._mem_miss = self._mem_miss[keep_mask]
        if self.enable_occlusion_prune:
            self._mem_u = self._mem_u[keep_mask]
            self._mem_v = self._mem_v[keep_mask]
            self._mem_zcam = self._mem_zcam[keep_mask]

    # ----------------------------
    # single-view occlusion prune (Femto)
    # ----------------------------
    def _occlusion_prune_memory_fast(self, now_xyz_base: np.ndarray):
        """
        Free-space carving:
            delete if z_mem < (z_now - margin)
        margin=0 => 가장 공격적(조금이라도 앞이면 삭제)
        """
        assert self.use_cuda
        if (not self.enable_occlusion_prune) or self._mem_keys.size == 0:
            return
        if now_xyz_base.size == 0:
            return

        u_now, v_now, z_now, _ = self._project_base_to_cam_torch(now_xyz_base)
        if u_now.size == 0:
            if self._mem_miss.size > 0:
                self._mem_miss = np.minimum(self._mem_miss + 1, np.int16(32767))
            return

        Z_full = self._rasterize_min_float_torch(u_now, v_now, z_now, self._depth_h, self._depth_w)

        k = 2 * self.occl_patch_radius + 1 if self.occl_patch_radius > 0 else 1
        Z_min = self._min_filter2d_float_torch(Z_full, k)

        valid_mem = (self._mem_u >= 0) & (self._mem_v >= 0) & (self._mem_zcam > 0.0)
        if not np.any(valid_mem):
            if self._mem_miss.size > 0:
                self._mem_miss = np.minimum(self._mem_miss + 1, np.int16(32767))
            return

        u_m = self._mem_u[valid_mem]
        v_m = self._mem_v[valid_mem]
        z_m = self._mem_zcam[valid_mem]          # meters
        z_now_at = Z_min[v_m, u_m].astype(np.float32, copy=False)

        # delete free-space (aggressive when margin=0)
        thr = z_now_at - float(self._free_space_margin_m)
        del_local_valid = (z_now_at > 0.0) & (z_m < thr)

        del_local_mask = np.zeros_like(valid_mem, dtype=bool)
        del_local_mask[np.where(valid_mem)[0][del_local_valid]] = True

        # miss update (hit if that pixel has current surface)
        hit_mask_global = np.zeros_like(valid_mem, dtype=bool)
        hit_mask_global[np.where(valid_mem)[0][z_now_at > 0.0]] = True
        self._mem_miss[hit_mask_global] = 0
        self._mem_miss[~hit_mask_global] = np.minimum(self._mem_miss[~hit_mask_global] + 1, np.int16(32767))

        age_all = (self._frame_idx - self._mem_step)
        del_by_miss = (self._mem_miss >= self._miss_prune_frames) & (age_all >= self._miss_min_age)

        if np.any(del_local_mask) or np.any(del_by_miss):
            keep_global = ~(del_local_mask | del_by_miss)
            self._delete_mask(keep_global)

    # ----------------------------
    # multiview helpers (D405 pinhole)
    # ----------------------------
    def _get_K_pinhole_torch(self, K_3x3: np.ndarray) -> torch.Tensor:
        """K_d405는 보통 고정이므로 torch 텐서 캐시."""
        K = np.asarray(K_3x3, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"K_3x3 must be (3,3), got {K.shape}")
        key = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        if self._K_pinhole_cache_key != key:
            self._K_pinhole_cache_key = key
            self._K_pinhole_cache_t = torch.from_numpy(K).to(self._device)
        return self._K_pinhole_cache_t

    def _project_base_to_cam_pinhole_torch(
        self,
        xyz_base_np: np.ndarray,
        base_to_cam_4x4: np.ndarray,
        K_3x3: np.ndarray,
        W: int,
        H: int,
    ):
        """
        rectified 가정(왜곡 없음).
        returns: u(int32), v(int32), z(float32 meters), orig_idx(int64)
        """
        if xyz_base_np.size == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f32 = np.array([], dtype=np.float32)
            empty_i64 = np.array([], dtype=np.int64)
            return empty_i32, empty_i32, empty_f32, empty_i64

        assert self.use_cuda
        dev = self._device

        xyz = torch.from_numpy(xyz_base_np.astype(np.float32, copy=False)).to(dev)
        ones = torch.ones((xyz.shape[0], 1), dtype=torch.float32, device=dev)
        xyz1 = torch.cat([xyz, ones], dim=1)

        T = torch.from_numpy(np.asarray(base_to_cam_4x4, dtype=np.float32)).to(dev)
        xyz_cam = (xyz1 @ T.T)[:, :3]

        z = xyz_cam[:, 2]
        valid = z > 0.0
        if torch.count_nonzero(valid) == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f32 = np.array([], dtype=np.float32)
            empty_i64 = np.array([], dtype=np.int64)
            return empty_i32, empty_i32, empty_f32, empty_i64

        pts = xyz_cam[valid]
        xn = pts[:, 0] / pts[:, 2]
        yn = pts[:, 1] / pts[:, 2]

        Kt = self._get_K_pinhole_torch(K_3x3)
        fx, fy = Kt[0, 0], Kt[1, 1]
        cx, cy = Kt[0, 2], Kt[1, 2]

        u = torch.round(fx * xn + cx).to(torch.int32)
        v = torch.round(fy * yn + cy).to(torch.int32)

        inb = (u >= 0) & (u < int(W)) & (v >= 0) & (v < int(H))
        if torch.count_nonzero(inb) == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f32 = np.array([], dtype=np.float32)
            empty_i64 = np.array([], dtype=np.int64)
            return empty_i32, empty_i32, empty_f32, empty_i64

        u = u[inb]
        v = v[inb]
        z_out = pts[inb, 2].to(torch.float32)
        orig_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)[inb]

        return (
            u.detach().cpu().numpy(),
            v.detach().cpu().numpy(),
            z_out.detach().cpu().numpy(),
            orig_idx.detach().cpu().numpy().astype(np.int64),
        )

    def _occlusion_prune_memory_multiview(
        self,
        *,
        Z_f: Optional[np.ndarray],        # (Hf,Wf) meters
        Z_d: Optional[np.ndarray],        # (Hd,Wd) meters
        base_to_cam_d: np.ndarray,        # D405 base->cam (매 프레임 갱신)
        K_d: np.ndarray,
        W_d: int,
        H_d: int,
    ):
        """
        miss 업데이트: hit_any = hit_f OR hit_d
        free-space delete(공격적): z_mem < (z_now - margin)
        """
        assert self.use_cuda
        if self._mem_keys.size == 0:
            return

        N = self._mem_xyz.shape[0]
        margin = float(self._free_space_margin_m)

        # ---------- Femto hit / delete ----------
        hit_f = np.zeros((N,), dtype=bool)
        del_f = np.zeros((N,), dtype=bool)

        if Z_f is not None and Z_f.size > 0 and self.enable_occlusion_prune:
            if Z_f.shape != (self._depth_h, self._depth_w):
                # shape mismatch면 조용히 skip 대신, 디버깅에 유리하게 에러
                raise ValueError(f"Z_f shape must be ({self._depth_h},{self._depth_w}), got {Z_f.shape}")

            valid_f = (self._mem_u >= 0) & (self._mem_v >= 0) & (self._mem_zcam > 0.0)
            if np.any(valid_f):
                u = self._mem_u[valid_f]
                v = self._mem_v[valid_f]
                z_m = self._mem_zcam[valid_f].astype(np.float32, copy=False)
                z_now = Z_f[v, u].astype(np.float32, copy=False)

                hit_local = (z_now > 0.0)
                hit_f[np.where(valid_f)[0][hit_local]] = True

                thr = z_now - margin
                del_local = (z_now > 0.0) & (z_m < thr)
                del_f[np.where(valid_f)[0][del_local]] = True

        # ---------- D405 hit / delete ----------
        hit_d = np.zeros((N,), dtype=bool)
        del_d = np.zeros((N,), dtype=bool)

        if Z_d is not None and Z_d.size > 0:
            if Z_d.shape != (int(H_d), int(W_d)):
                raise ValueError(f"Z_d shape must be ({H_d},{W_d}), got {Z_d.shape}")

            u_d, v_d, z_d, idx = self._project_base_to_cam_pinhole_torch(
                self._mem_xyz, base_to_cam_d, K_d, int(W_d), int(H_d)
            )
            if idx.size > 0:
                z_now_d = Z_d[v_d, u_d].astype(np.float32, copy=False)

                hit_local = (z_now_d > 0.0)
                hit_d[idx[hit_local]] = True

                thr = z_now_d - margin
                del_local = (z_now_d > 0.0) & (z_d.astype(np.float32, copy=False) < thr)
                del_d[idx[del_local]] = True

        # ---------- miss update: hit_any ----------
        hit_any = hit_f | hit_d
        self._mem_miss[hit_any] = 0
        self._mem_miss[~hit_any] = np.minimum(self._mem_miss[~hit_any] + 1, np.int16(32767))

        # ---------- miss-based delete ----------
        age_all = (self._frame_idx - self._mem_step)
        del_by_miss = (self._mem_miss >= self._miss_prune_frames) & (age_all >= self._miss_min_age)

        # ---------- final delete ----------
        del_free_space = del_f | del_d
        if np.any(del_free_space) or np.any(del_by_miss):
            keep = ~(del_free_space | del_by_miss)
            self._delete_mask(keep)

    # ----------------------------
    # public: process / process_wrist / process_global
    # ----------------------------
    def process(self, points):
        if points is None or len(points) == 0:
            if self.enable_temporal and self.export_mode == 'fused':
                self._frame_idx += 1
                return np.zeros((0, 7), dtype=np.float32)
            return np.zeros((self.target_num_points, 6), dtype=np.float32)

        points = points.astype(np.float32, copy=False)

        if self.enable_transform:
            points = self._apply_transform(points)
        if self.enable_cropping:
            points = self._crop_workspace(points)
        if self.enable_filter:
            points = self._apply_variance_filter(points)

        if (not self.enable_temporal) or (self.export_mode != "fused"):
            if self.enable_sampling:
                points = self._sample_points(points)
            return points

        now_step = self._frame_idx

        if self.enable_occlusion_prune:
            self._occlusion_prune_memory_fast(points[:, :3])

        xyz_now = points[:, :3]
        rgb_now = points[:, 3:6]

        xyz_now, rgb_now, keys_new = self._frame_unique_torch(xyz_now, rgb_now)
        self._merge_into_mem(xyz_now, rgb_now, now_step, keys_new=keys_new)
        self._prune_mem(now_step)

        out = self._export_array_from_mem(now_step)
        self._frame_idx += 1
        return out

    def process_wrist(self, points, dynamic_extrinsics=None, dynamic_extrinsics_is_cam_to_base: bool = True):
        """
        wrist(D405) 포인트클라우드용.
        - dynamic_extrinsics: (4,4)
          기본은 cam_to_base로 가정.
          만약 base_to_cam을 넘긴다면 dynamic_extrinsics_is_cam_to_base=False 로 호출.
        """
        if points is None or len(points) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        points = points.astype(np.float32, copy=False)
        if self.enable_transform and dynamic_extrinsics is not None:
            T = np.asarray(dynamic_extrinsics, dtype=np.float32)
            if T.shape != (4, 4):
                raise ValueError(f"dynamic_extrinsics must be (4,4), got {T.shape}")
            if not dynamic_extrinsics_is_cam_to_base:
                T = np.linalg.inv(T).astype(np.float32)
            points = self._apply_dynamic_transform(points, T)
        return points

    def process_global(
        self,
        *,
        femto_points: np.ndarray,         # (Nf,6) (process 입력과 동일)
        femto_depth_u16: np.ndarray,      # (Hf,Wf) uint16
        d405_points_base: np.ndarray,     # (Nd,6) 이미 base로 TF된 D405 PC
        d405_depth_u16: np.ndarray,       # (Hd,Wd) uint16
        d405_cam_to_base: np.ndarray,     # (4,4) cam->base (동적)
        K_d405: np.ndarray,               # (3,3)
        d405_wh: tuple,                   # (W,H)
        femto_z_unit: str = "mm",
        d405_z_unit: str = "0.1mm",
    ) -> np.ndarray:
        """
        - (A) Femto PC는 기존 process와 동일하게 TF/crop/filter
        - (B) D405 PC는 base로 들어온다고 가정하고 crop만 적용
        - (C) Z-buffer는 depth_u16 자체로 Zmin 생성 (rasterize 안씀, 빠름)
        - (D) multiview prune: hit_any miss + free-space delete
        - (E) 현재 프레임(Femto+D405) merge 후 temporal export
        """
        if not (self.enable_temporal and self.export_mode == "fused"):
            raise RuntimeError("process_global은 temporal fused 모드에서만 사용하세요.")
        assert self.use_cuda

        # (1) Femto points preprocess
        pts_f = np.asarray(femto_points, dtype=np.float32)
        if pts_f.size > 0:
            if self.enable_transform:
                pts_f = self._apply_transform(pts_f)
            if self.enable_cropping:
                pts_f = self._crop_workspace(pts_f)
            if self.enable_filter:
                pts_f = self._apply_variance_filter(pts_f)

        # (2) D405 points (already base)
        pts_d = np.asarray(d405_points_base, dtype=np.float32)
        if pts_d.size > 0 and self.enable_cropping:
            pts_d = self._crop_workspace(pts_d)

        # (3) depth -> Zmin (meters)
        Z_f = self._zmin_from_depth_u16(femto_depth_u16, z_unit=femto_z_unit, patch_radius=self.occl_patch_radius)
        Z_d = self._zmin_from_depth_u16(d405_depth_u16, z_unit=d405_z_unit, patch_radius=self.occl_patch_radius)

        # (4) D405 base->cam
        cam_to_base = np.asarray(d405_cam_to_base, dtype=np.float32)
        if cam_to_base.shape != (4, 4):
            raise ValueError(f"d405_cam_to_base must be (4,4), got {cam_to_base.shape}")
        base_to_cam_d = np.linalg.inv(cam_to_base).astype(np.float32)

        Wd, Hd = int(d405_wh[0]), int(d405_wh[1])

        # (5) multiview prune/miss update
        self._occlusion_prune_memory_multiview(
            Z_f=Z_f,
            Z_d=Z_d,
            base_to_cam_d=base_to_cam_d,
            K_d=np.asarray(K_d405, dtype=np.float32),
            W_d=Wd,
            H_d=Hd,
        )

        # (6) merge current frame to global
        if pts_f.size == 0 and pts_d.size == 0:
            now_step = self._frame_idx
            out = self._export_array_from_mem(now_step)
            self._frame_idx += 1
            return out

        if pts_f.size and pts_d.size:
            pts_now = np.concatenate([pts_f, pts_d], axis=0)
        else:
            pts_now = pts_f if pts_f.size else pts_d

        xyz_now = pts_now[:, :3]
        rgb_now = pts_now[:, 3:6]

        now_step = self._frame_idx
        xyz_now, rgb_now, keys_new = self._frame_unique_torch(xyz_now, rgb_now)
        self._merge_into_mem(xyz_now, rgb_now, now_step, keys_new=keys_new)
        self._prune_mem(now_step)

        out = self._export_array_from_mem(now_step)
        self._frame_idx += 1
        return out

    # ----------------------------
    # geometry preprocess
    # ----------------------------
    def _apply_transform(self, points):
        # NOTE: 기존 코드 그대로 유지 (입력 xyz가 mm라고 가정하고 0.001)
        points = points[points[:, 2] > 0.0]
        if len(points) == 0:
            return points

        point_xyz = points[:, :3] * 0.001  # mm -> m
        point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1), dtype=point_xyz.dtype)))
        point_transformed = np.dot(point_homogeneous, self.extrinsics_matrix.T)

        points[:, :3] = point_transformed[:, :3].astype(np.float32, copy=False)
        return points

    def _apply_dynamic_transform(self, points, dynamic_extrinsics):
        points = points[points[:, 2] > 0.0]
        if len(points) == 0:
            return points
        point_xyz = points[:, :3]
        point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1), dtype=point_xyz.dtype)))
        point_transformed = np.dot(point_homogeneous, dynamic_extrinsics.T)
        points[:, :3] = point_transformed[:, :3]
        return points

    def _crop_workspace(self, points):
        if len(points) == 0:
            return points
        mask = (
            (points[:, 0] >= self.workspace_bounds[0][0]) &
            (points[:, 0] <= self.workspace_bounds[0][1]) &
            (points[:, 1] >= self.workspace_bounds[1][0]) &
            (points[:, 1] <= self.workspace_bounds[1][1]) &
            (points[:, 2] >= self.workspace_bounds[2][0]) &
            (points[:, 2] <= self.workspace_bounds[2][1])
        )
        return points[mask]

    # ----------------------------
    # variance filter / sampling (기존 유지)
    # ----------------------------
    def _apply_variance_filter(self, points):
        if len(points) == 0:
            raise ValueError("points empty")

        xyz = points[:, :3]
        u, v, z, idx = self._project_base_to_cam_torch(xyz)
        if len(u) == 0:
            return points

        Z = self._rasterize_min_float_torch(u, v, z, self._depth_h, self._depth_w)
        kernel_size = int(self.variance_kernel_size)
        variance_threshold = float(self.variance_threshold) * 1e-6  # mm^2 -> m^2

        valid_mask = Z > 0
        depth_zero_filled = np.where(valid_mask, Z, 0).astype(np.float32, copy=False)

        ksize = (kernel_size, kernel_size)
        depth_sum = cv2.boxFilter(depth_zero_filled, -1, ksize, normalize=False)
        valid_count = cv2.boxFilter(valid_mask.astype(np.float32), -1, ksize, normalize=False)
        valid_count_safe = np.maximum(valid_count, 1.0)

        mean = depth_sum / valid_count_safe
        depth_sq_sum = cv2.boxFilter(depth_zero_filled ** 2, -1, ksize, normalize=False)
        sq_mean = depth_sq_sum / valid_count_safe
        variance = sq_mean - mean ** 2

        keep_mask_2d = (variance < variance_threshold) & valid_mask

        u_idx = np.clip(np.round(u).astype(int), 0, self._depth_w - 1)
        v_idx = np.clip(np.round(v).astype(int), 0, self._depth_h - 1)

        projected_keep = keep_mask_2d[v_idx, u_idx] & (np.abs(z - Z[v_idx, u_idx]) < 1e-3)
        kept_original_indices = idx[projected_keep]
        return points[kept_original_indices]

    def _sample_points(self, points):
        if len(points) == 0:
            return np.zeros((self.target_num_points, 6), dtype=np.float32)

        if len(points) <= self.target_num_points:
            padded = np.zeros((self.target_num_points, 6), dtype=np.float32)
            padded[:len(points)] = points
            return padded

        try:
            points_xyz = points[:, :3]
            sampled_xyz, sample_indices = self._farthest_point_sampling(points_xyz, self.target_num_points)
            if self.use_cuda:
                sample_indices = sample_indices.cpu().numpy().flatten()
            else:
                sample_indices = sample_indices.numpy().flatten()
            return points[sample_indices]
        except Exception:
            indices = np.random.choice(len(points), self.target_num_points, replace=False)
            return points[indices]

    def _farthest_point_sampling(self, points, num_points):
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("pytorch3d not available")
        points_tensor = torch.from_numpy(points)
        if self.use_cuda:
            points_tensor = points_tensor.cuda()
        points_batch = points_tensor.unsqueeze(0)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points_batch, K=num_points)
        sampled_points = sampled_points.squeeze(0)
        indices = indices.squeeze(0)
        if self.use_cuda:
            sampled_points = sampled_points.cpu()
        return sampled_points.numpy(), indices
