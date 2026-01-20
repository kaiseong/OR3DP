# real_data_pc_conversion.py



from typing import Sequence, Tuple, Dict, Optional, Union, List
import os
import pathlib
import cv2
import numpy as np
import zarr
import numcodecs
from tqdm import tqdm
import open3d as o3d
import torch
import pcdp.common.mono_time as mono_time
try:
    import pytorch3d.ops as torch3d_ops
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: pytorch3d not available. FPS will use fallback method.")


from pcdp.common import RISE_transformation as rise_tf
from pcdp.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from pcdp.common.RISE_transformation import xyz_rot_transform

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])

class PointCloudPreprocessor:


    @staticmethod
    def _is_orbbec_intrinsics(obj) -> bool:
        return all(hasattr(obj, a) for a in ("fx", "fy", "cx", "cy"))

    @staticmethod
    def _is_orbbec_distortion(obj) -> bool:
        return all(hasattr(obj, a) for a in ("k1","k2","k3","k4","k5","k6","p1","p2"))

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
    # -------------------------------------------------

    def __init__(self, 
                enable_sampling=False,
                target_num_points=1024,
                enable_transform=True,
                extrinsics_matrix=None,
                enable_cropping=True,
                workspace_bounds=None,
                enable_filter=False,
                nb_points=10,
                sor_std=1.7,
                use_cuda=True,
                verbose=False,
                enable_temporal=False,
                export_mode='off',
                temporal_voxel_size=0.005,
                temporal_decay=0.96,
                temporal_c_min=0.20,
                temporal_prune_every: int=1,
                stable_export: bool = False,
                enable_occlusion_prune: bool = True,
                depth_width: Optional[int] = 320,
                depth_height: Optional[int] = 288,
                K_depth: Optional[Sequence[Sequence[float]]] = None,    
                dist_depth: Optional[Sequence[float]] = None,           
                erode_k: int = 1,
                z_unit: str = 'm',   # 'm' or 'mm'
                occl_patch_radius: int = 2,
                miss_prune_frames: int = 20,   
                miss_min_age: int = 2         
                ):

        if extrinsics_matrix is None:
            self.extrinsics_matrix = np.array([
                [  0.007131,  -0.91491,    0.403594,  0.05116],
                [ -0.994138,   0.003833,   0.02656,  -0.00918],
                [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
                [  0.,         0. ,        0. ,        1.      ]
            ])
        else:
            self.extrinsics_matrix = np.array(extrinsics_matrix)
            
        # Default workspace bounds
        if workspace_bounds is None:
            self.workspace_bounds = [
                [0.132, 0.715],    
                [-0.400, 0.350],   
                [-0.100, 0.400]    
            ]
        else:
            self.workspace_bounds = workspace_bounds
        
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

        if dist_depth is None:
            self._dist_depth = np.array(
                [11.690222, 5.343991, 0.000065, 0.000014, 0.172997, 12.017323, 9.254467, 1.165690],
                dtype=np.float64)
        else:
            if self._is_orbbec_distortion(dist_depth):
                self._dist_depth = self._as_cv_dist_from_orbbec_distortion(dist_depth)
            else:
                self._dist_depth = np.asarray(dist_depth, dtype=np.float64)
                if self._dist_depth.size not in (4, 5, 8):
                    raise ValueError(f"dist_depth must have 4/5/8 coeffs, got {self._dist_depth.size}")

        self.target_num_points = target_num_points
        self.nb_points = nb_points
        self.sor_std = sor_std
        self.enable_transform = enable_transform
        self.enable_cropping = enable_cropping
        self.enable_sampling = enable_sampling
        self.enable_filter = enable_filter
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.verbose = verbose
        
        self.enable_temporal = enable_temporal
        self.export_mode = export_mode
        self._temporal_voxel_size = float(temporal_voxel_size)
        self._temporal_decay = float(temporal_decay)
        self._temporal_c_min = float(temporal_c_min)
        self._prune_every = int(max(1, temporal_prune_every))
        self._stable_export = bool(stable_export)
            
        self.enable_occlusion_prune = bool(enable_occlusion_prune)
        self._depth_w = int(depth_width) 
        self._depth_h = int(depth_height) 
        self._erode_k = int(erode_k)
        self._z_unit = str(z_unit)
        self.occl_patch_radius = int(occl_patch_radius)
        

        self._miss_prune_frames = int(miss_prune_frames)
        self._miss_min_age = int(miss_min_age)

        self._base_to_cam = None
        if self.enable_occlusion_prune:
            if self.extrinsics_matrix is None:
                raise ValueError("enable_occlusion_prune=True인데 extrinsics_matrix가 없습니다.")
            self._base_to_cam = np.linalg.inv(np.array(self.extrinsics_matrix, dtype=np.float64))
            if self._K_depth is None or self._depth_w is None or self._depth_h is None:
                raise ValueError("occlusion prune에는 depth_width/height와 K_depth가 필요합니다.")

        self._frame_idx = 0
        self._mem_keys = np.empty((0,), dtype=np.dtype((np.void, 12)))
        self._mem_xyz = np.empty((0, 3), dtype=np.float32)
        self._mem_rgb = np.empty((0, 3), dtype=np.float32)
        self._mem_step = np.empty((0, ), dtype=np.int32)
        self._mem_miss = np.empty((0, ), dtype=np.int16)  

        # 
        self._mem_u    = np.empty((0,), dtype=np.int32)   
        self._mem_v    = np.empty((0,), dtype=np.int32)   
        self._mem_zcam = np.empty((0,), dtype=np.float32) 


        if 0.0 < self._temporal_decay < 1.0 and 0.0 < self._temporal_c_min < 1.0:
            self._max_age_steps = int(np.floor(np.log(self._temporal_c_min)/np.log(self._temporal_decay)))
        else:
            self._max_age_steps = 0

        if self.verbose:
            print(f"PointCloudPreprocessor initialized:")
            print(f"  - Transform: {self.enable_transform}")
            print(f"  - Cropping: {self.enable_cropping}")
            print(f"  - Sampling: {self.enable_sampling} (target: {self.target_num_points})")
            print(f"  - CUDA: {self.use_cuda}")
        
        if (self.enable_temporal and self.export_mode == 'fused') or self.enable_occlusion_prune:
            if not self.use_cuda:
                raise RuntimeError(
                    "GPU-only path enabled (temporal fused / occlusion prune). "
                    "Set use_cuda=True and ensure CUDA is available."
                )
            self._maybe_init_torch_camera()

    def __call__(self, points):
        return self.process(points)

    def _erode_min(self, depth_u16: np.ndarray):
        k = self._erode_k
        if k <= 0: 
            return depth_u16
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k+1, 2*k+1))
        d = depth_u16.copy()
        zero = (d == 0)
        d[zero] = 65535
        d = cv2.erode(d, kernel)
        d[zero] = 0
        return d

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

        if getattr(self, "_base_to_cam", None) is not None:
            self._base_to_cam_t = torch.tensor(self._base_to_cam.astype(np.float32), device=dev)
        else:
            self._base_to_cam_t = None

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
            empty_int = np.array([], dtype=np.int64)  
            return empty_i32, empty_i32, empty_f64, empty_int


        z = xyz_cam_t[:, 2]
        valid = z > 0.0
        if torch.count_nonzero(valid) == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            empty_int = np.array([], dtype=np.int64)  
            return empty_i32, empty_i32, empty_f64, empty_int

        pts = xyz_cam_t[valid]  # (M,3)
        xn = pts[:, 0] / pts[:, 2]
        yn = pts[:, 1] / pts[:, 2]

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
            empty_int = np.array([], dtype=np.int64)  
            return empty_i32, empty_i32, empty_f64, empty_int

        u = u[inb]
        v = v[inb]
        z_out = pts[inb, 2]

        orig_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)[inb]

        orig_idx_np = orig_idx.detach().cpu().numpy().astype(np.int64)  
        return (
            u.detach().cpu().numpy(),
            v.detach().cpu().numpy(),
            z_out.detach().cpu().numpy().astype(np.float64),
            orig_idx_np
        )
        

    def _project_base_to_cam_torch(self, xyz_base_np: np.ndarray):
        if xyz_base_np.size == 0:
            empty_i32 = np.array([], dtype=np.int32)
            empty_f64 = np.array([], dtype=np.float64)
            empty_int = np.array([], dtype=np.int64)  
            return empty_i32, empty_i32, empty_f64, empty_int

        self._maybe_init_torch_camera()
        assert self._base_to_cam_t is not None, \
            "Base→Cam not initialized. Provide extrinsics_matrix and enable_occlusion_prune=True."

        xyz = torch.from_numpy(xyz_base_np.astype(np.float32, copy=False)).to(self._device)
        ones = torch.ones((xyz.shape[0], 1), dtype=torch.float32, device=self._device)
        xyz1 = torch.cat([xyz, ones], dim=1)  
        xyz_cam = (xyz1 @ self._base_to_cam_t.T)[:, :3]  
        return self._project_points_cam_torch(xyz_cam)


    def reset_temporal(self):
        self._frame_idx = 0
        self._mem_keys = np.empty((0, ), dtype=np.dtype((np.void, 12)))
        self._mem_xyz   = np.empty((0, 3), dtype=np.float32)
        self._mem_rgb   = np.empty((0, 3), dtype=np.float32)
        self._mem_step  = np.empty((0,),  dtype=np.int32)
        self._mem_miss  = np.empty((0,),  dtype=np.int16)

        self._mem_u    = np.empty((0,), dtype=np.int32)
        self._mem_v    = np.empty((0,), dtype=np.int32)
        self._mem_zcam = np.empty((0,), dtype=np.float32)

    def voxel_keys_from_xyz(self, xyz:np.ndarray):
        grid = np.floor(xyz / self._temporal_voxel_size).astype(np.int32, copy=False) 
        grid = np.ascontiguousarray(grid)
        keys = grid.view(np.dtype((np.void, 12))).ravel() 
        return keys

    def _frame_unique_torch(self, xyz_np: np.ndarray, rgb_np: np.ndarray, last_wins: bool=False):
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
            self._mem_xyz  = xyz_new.astype(np.float32, copy=False).copy()
            self._mem_rgb  = rgb_new.astype(np.float32, copy=False).copy()
            self._mem_step = np.full((xyz_new.shape[0],), now_step, dtype=np.int32)
            self._mem_miss = np.zeros((xyz_new.shape[0],), dtype=np.int16)
            if self.enable_occlusion_prune:
                u_add, v_add, z_add = self._project_and_pack(self._mem_xyz)
                self._mem_u, self._mem_v, self._mem_zcam = u_add, v_add, z_add
            else:
                self._mem_u    = np.empty((0,), dtype=np.int32)
                self._mem_v    = np.empty((0,), dtype=np.int32)
                self._mem_zcam = np.empty((0,), dtype=np.float32)
            return
        
        common, idx_mem, idx_new = np.intersect1d(self._mem_keys, keys_new, return_indices=True)
        if common.size > 0:
            self._mem_xyz[idx_mem]  = xyz_new[idx_new]
            self._mem_rgb[idx_mem]  = rgb_new[idx_new]
            self._mem_step[idx_mem] = now_step
            self._mem_miss[idx_mem] = 0
            if self.enable_occlusion_prune:
                u_upd, v_upd, z_upd = self._project_and_pack(xyz_new[idx_new])
                self._mem_u[idx_mem]    = u_upd
                self._mem_v[idx_mem]    = v_upd
                self._mem_zcam[idx_mem] = z_upd

        mask_new_only = np.ones(keys_new.shape[0], dtype=bool)
        if common.size > 0:
            mask_new_only[idx_new] = False
        if mask_new_only.any():
            add_xyz  = xyz_new[mask_new_only]        
            add_rgb  = rgb_new[mask_new_only]
            add_keys = keys_new[mask_new_only]
            self._mem_keys = np.concatenate([self._mem_keys, add_keys], axis=0)
            self._mem_xyz  = np.concatenate([self._mem_xyz,  add_xyz], axis=0)
            self._mem_rgb  = np.concatenate([self._mem_rgb,  add_rgb], axis=0)
            self._mem_step = np.concatenate([self._mem_step,
                                             np.full((add_xyz.shape[0],), now_step, dtype=np.int32)], axis=0)
            self._mem_miss = np.concatenate([self._mem_miss,
                                             np.zeros((add_xyz.shape[0],), dtype=np.int16)], axis=0)
            if self.enable_occlusion_prune: 
                u_add, v_add, z_add = self._project_and_pack(add_xyz)
                self._mem_u    = np.concatenate([self._mem_u,    u_add], axis=0)
                self._mem_v    = np.concatenate([self._mem_v,    v_add], axis=0)
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
            return np.zeros((0,7), dtype=np.float32)
        age = (now_step - self._mem_step).astype(np.float32)
        c   = (self._temporal_decay** age).astype(np.float32, copy=False)  # (N,)

        if self._stable_export:
            order = np.argsort(self._mem_keys)
            xyz = self._mem_xyz[order]
            rgb = self._mem_rgb[order]
            c   = c[order]
        else:
            xyz = self._mem_xyz
            rgb = self._mem_rgb

        out = np.concatenate([xyz, rgb, c[:,None]], axis=1).astype(np.float32, copy=False)
        return out


    def _rasterize_min_float_torch(self, u: np.ndarray, v: np.ndarray, z_m: np.ndarray, H: int, W: int):
        assert self.use_cuda, "GPU-only path: _rasterize_min_float_torch requires CUDA"
        if u.size == 0:
            return np.zeros((H, W), dtype=np.float32)

        device = self._device 
        pix = torch.from_numpy((v * W + u).astype(np.int64)).to(device, non_blocking=True)
        zt  = torch.from_numpy(z_m.astype(np.float32, copy=False)).to(device, non_blocking=True)

        Z = torch.full((H * W,), float("inf"), device=device, dtype=torch.float32)
        try:
            Z = torch.scatter_reduce(Z, 0, pix, zt, reduce='amin', include_self=True)
        except TypeError:
            Z.scatter_reduce_(0, pix, zt, reduce='amin', include_self=True)

        Z = Z.view(H, W)
        Z[torch.isinf(Z)] = 0.0
        return Z.detach().cpu().numpy()


    def _min_filter2d_float_torch(self, Z: np.ndarray, k: int):
        assert self.use_cuda, "GPU-only path: _min_filter2d_float_torch requires CUDA"
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



    def _project_and_pack(self, xyz_base: np.ndarray):
        assert self.use_cuda, "GPU-only path: _project_and_pack requires CUDA"
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
        self._mem_xyz  = self._mem_xyz[keep_mask]
        self._mem_rgb  = self._mem_rgb[keep_mask]
        self._mem_step = self._mem_step[keep_mask]
        self._mem_miss = self._mem_miss[keep_mask]
        if self.enable_occlusion_prune:
            self._mem_u    = self._mem_u[keep_mask]
            self._mem_v    = self._mem_v[keep_mask]
            self._mem_zcam = self._mem_zcam[keep_mask]

    def _occlusion_prune_memory_fast(self, now_xyz_base: np.ndarray):
        assert self.use_cuda, "GPU-only path: _occlusion_prune_memory_fast requires CUDA"
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
        z_m = self._mem_zcam[valid_mem]  # meters
        z_now_at = Z_min[v_m, u_m]       # meters

        del_local_valid = (z_now_at > 0.0) & (z_m < z_now_at)
        del_local_mask = np.zeros_like(valid_mem, dtype=bool)
        del_local_mask[np.where(valid_mem)[0][del_local_valid]] = True

        hit_mask_global = np.zeros_like(valid_mem, dtype=bool)
        hit_mask_global[np.where(valid_mem)[0][z_now_at > 0.0]] = True
        self._mem_miss[hit_mask_global] = 0
        self._mem_miss[~hit_mask_global] = np.minimum(self._mem_miss[~hit_mask_global] + 1, np.int16(32767))

        age_all = (self._frame_idx - self._mem_step)
        del_by_miss = (self._mem_miss >= self._miss_prune_frames) & (age_all >= self._miss_min_age)

        if np.any(del_local_mask) or np.any(del_by_miss):
            keep_global = ~(del_local_mask | del_by_miss)
            self._delete_mask(keep_global)


    def process(self, points):
        if points is None or len(points) == 0:
            if self.enable_temporal and self.export_mode == 'fused':
                self._frame_idx += 1
                return np.zeros((0,7), dtype=np.float32)
            return np.zeros((self.target_num_points, 6), dtype=np.float32)
            
        points = points.astype(np.float32)
        if self.enable_transform:
            points = self._apply_transform(points)
        if self.enable_cropping:
            points = self._crop_workspace(points)
        if self.enable_filter:
            points = self._apply_filter(points)
        if not self.enable_temporal or self.export_mode!="fused":
            if self.enable_sampling:
                points = self._sample_points(points)
            return points
        

        now_step = self._frame_idx

        xyz_now = points[:, :3]
        rgb_now = points[:, 3:6]

        if self.enable_occlusion_prune:
            self._occlusion_prune_memory_fast(points[:, :3])
        
        assert self.use_cuda, "GPU-only path: fused temporal export requires CUDA"
        xyz_now, rgb_now, keys_new = self._frame_unique_torch(xyz_now, rgb_now)
        self._merge_into_mem(xyz_now, rgb_now, now_step, keys_new=keys_new)


        self._prune_mem(now_step)

        out = self._export_array_from_mem(now_step)
        
        
        self._frame_idx += 1
        return out


    def _apply_transform(self, points):

        points = points[points[:, 2] > 0.0]
        point_xyz = points[:, :3] * 0.001
        
        point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
        point_transformed = np.dot(point_homogeneous, self.extrinsics_matrix.T)
        
        points[:, :3] = point_transformed[:, :3]
        
        if self.verbose and len(points) > 0:
            print(f"After transform: {len(points)} points, "
                  f"XYZ range: [{points[:, :3].min(axis=0)} - {points[:, :3].max(axis=0)}]")
        
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
        
        cropped_points = points[mask]
        
        if self.verbose:
            print(f"After cropping: {len(cropped_points)}/{len(points)} points remain")
            
        return cropped_points

    def _apply_filter(self, points):
        if len(points) == 0:
            raise ValueError("points empty")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # _, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_points, std_ratio=self.sor_std)
        _, ind = pcd.remove_radius_outlier(nb_points=12, radius=0.01)
        return points[ind]




    def _sample_points(self, points):
        if len(points) == 0:
            return np.zeros((self.target_num_points, 6), dtype=np.float32)
            
        if len(points) <= self.target_num_points:
            padded_points = np.zeros((self.target_num_points, 6), dtype=np.float32)
            padded_points[:len(points)] = points
            return padded_points
            
        try:
            points_xyz = points[:, :3]
            sampled_xyz, sample_indices = self._farthest_point_sampling(
                points_xyz, self.target_num_points)
                
            if self.use_cuda:
                sample_indices = sample_indices.cpu().numpy().flatten()
            else:
                sample_indices = sample_indices.numpy().flatten()
                
            sampled_points = points[sample_indices]
            
            if self.verbose:
                print(f"After sampling: {len(sampled_points)} points")
                
            return sampled_points
            
        except Exception as e:
            if self.verbose:
                print(f"FPS failed: {e}, using random sampling")
            indices = np.random.choice(len(points), self.target_num_points, replace=False)
            return points[indices]
            
    def _farthest_point_sampling(self, points, num_points):
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("pytorch3d not available")
            
        points_tensor = torch.from_numpy(points)
        
        if self.use_cuda:
            points_tensor = points_tensor.cuda()
            
        points_batch = points_tensor.unsqueeze(0)
        
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_batch, K=num_points)
            
        sampled_points = sampled_points.squeeze(0)
        indices = indices.squeeze(0)
        
        if self.use_cuda:
            sampled_points = sampled_points.cpu()
            
        return sampled_points.numpy(), indices

class LowDimPreprocessor:
    def __init__(self,
                 robot_to_base=None
                ):
        if robot_to_base is None:
            self.robot_to_base=np.array([
                [1., 0., 0., -0.04],
                [0., 1., 0., -0.29],
                [0., 0., 1., -0.03],
                [0., 0., 0.,  1.0]
            ])
        else:
            self.robot_to_base = np.array(robot_to_base, dtype=np.float32)
        
    
    def TF_process(self, robot_7ds):
        assert robot_7ds.shape[-1] == 7, f"robot_7ds data shape shoud be (..., 7), but got {robot_7ds.shape}"
        processed_robot7d = []
        for robot_7d in robot_7ds:
            pose_6d = robot_7d[:6]
            gripper = robot_7d[6]
            
            translation = pose_6d[:3]
            rotation = pose_6d[3:6]
            eef_to_robot_base_k = rise_tf.rot_trans_mat(translation, rotation)
            T_k_matrix = self.robot_to_base @ eef_to_robot_base_k
            transformed_pose_6d = rise_tf.mat_to_xyz_rot(
                T_k_matrix,
                rotation_rep='euler_angles',
                rotation_rep_convention='ZYX'
            )
            new_robot_7d = np.concatenate([transformed_pose_6d, [gripper]])
            processed_robot7d.append(new_robot_7d)
        
        return np.array(processed_robot7d, dtype=np.float32)




def create_default_preprocessor(target_num_points=1024, use_cuda=True, verbose=False):
    return PointCloudPreprocessor(
        target_num_points=target_num_points,
        use_cuda=use_cuda,
        verbose=verbose
    )


def downsample_obs_data(obs_data, downsample_factor=3, offset=0):


    downsampled_data = {}
    for key, value in obs_data.items():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            assert 0 <= offset < downsample_factor, "offset out of range"
            downsampled_data[key] = value[offset::downsample_factor].copy()
        else:
            downsampled_data[key] = value

    return downsampled_data


def align_obs_action_data(obs_data, action_data, obs_timestamps, action_timestamps):

    valid_indices = []
    aligned_action_indices = []
    
    for i, obs_ts in enumerate(obs_timestamps):
        future_actions = action_timestamps >= obs_ts
        if np.any(future_actions):
            action_idx = np.where(future_actions)[0][0]
            valid_indices.append(i)
            aligned_action_indices.append(action_idx)
    
    if len(valid_indices) == 0:
        print("Warning: No valid obs-action alignments found!")
        return {}, {}, []
    
    aligned_obs_data = {}
    for key, value in obs_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            aligned_obs_data[key] = value[valid_indices]
        else:
            aligned_obs_data[key] = value
            
    aligned_action_data = {}
    for key, value in action_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            aligned_action_data[key] = value[aligned_action_indices]
        else:
            aligned_action_data[key] = value
    return aligned_obs_data, aligned_action_data, valid_indices



def process_single_episode(episode_path, pc_preprocessor=None, lowdim_preprocessor=None, 
                            downsample_factor=3, downsample_offset=0):

    episode_path = pathlib.Path(episode_path)
    if pc_preprocessor is not None and hasattr(pc_preprocessor, "reset_temporal"):
        pc_preprocessor.reset_temporal()
    
    obs_zarr_path = episode_path / 'obs_replay_buffer.zarr'
    action_zarr_path = episode_path / 'action_replay_buffer.zarr'
    
    if not obs_zarr_path.exists() or not action_zarr_path.exists():
        raise FileNotFoundError(f"Missing zarr files in {episode_path}")
    
    obs_replay_buffer = ReplayBuffer.create_from_path(str(obs_zarr_path), mode='r')
    action_replay_buffer = ReplayBuffer.create_from_path(str(action_zarr_path), mode='r')

    obs_data ={}
    for key in obs_replay_buffer.keys():
        obs_data[key] = obs_replay_buffer[key][:]
    
    action_data ={}
    for key in action_replay_buffer.keys():
        action_data[key] = action_replay_buffer[key][:]

    downsampled_obs = downsample_obs_data(obs_data, downsample_factor=downsample_factor, offset=downsample_offset)
    downsampled_obs_timestamps = downsampled_obs['align_timestamp']
    action_timestamps = action_data['timestamp']
    
    aligned_obs, aligned_action, valid_indices = align_obs_action_data(
        downsampled_obs, action_data, 
        downsampled_obs_timestamps, action_timestamps)
    
    
    if len(valid_indices) == 0:
        return None
        
    if pc_preprocessor is not None and 'pointcloud' in aligned_obs:
        processed_pointclouds = []
        for pc in aligned_obs['pointcloud']:
            processed_pc = pc_preprocessor.process(pc)
            processed_pointclouds.append(processed_pc)
        aligned_obs['pointcloud'] = np.array(processed_pointclouds, dtype=object)
    
    
    robot_eef_pose = aligned_obs['robot_eef_pose']
    robot_gripper_width = aligned_obs['robot_gripper'][:, :1] 
    aligned_obs['robot_obs'] = np.concatenate([robot_eef_pose, robot_gripper_width], axis=1) 
    

    if lowdim_preprocessor is not None:
        aligned_obs['robot_obs'] = lowdim_preprocessor.TF_process(aligned_obs['robot_obs'])
        aligned_action['action'] = lowdim_preprocessor.TF_process(aligned_action['action'])
    
    episode_data = {}
    episode_data.update(aligned_obs)
    episode_data.update(aligned_action)
    
    return episode_data


def parse_shape_meta(shape_meta: dict) -> Tuple[List[str], List[str], dict, dict]:
    pointcloud_keys = []
    lowdim_keys = []
    pointcloud_configs = {}
    lowdim_configs = {}
    
    obs_shape_meta = shape_meta.get('obs', {})
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape', []))
        
        if obs_type == 'pointcloud':
            pointcloud_keys.append(key)
            pointcloud_configs[key] = {
                'shape': shape,
                'type': obs_type
            }
        elif obs_type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_configs[key] = {
                'shape': shape,
                'type': obs_type
            }
    
    return pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs


def validate_episode_data_with_shape_meta(episode_data: dict, shape_meta: dict) -> bool:

    pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs = parse_shape_meta(shape_meta)
    
    for key in pointcloud_keys:
        if key in episode_data:
            data = episode_data[key]
            expected_shape = pointcloud_configs[key]['shape']
            if len(data.shape) >= 2:
                if data.shape[-len(expected_shape):] != expected_shape:
                    print(f"Warning: {key} shape mismatch. Expected: {expected_shape}, Got: {data.shape}")
                    return False
        else:
            print(f"Warning: Expected pointcloud key '{key}' not found in episode data")
            return False
    
    for key in lowdim_keys:
        if key in episode_data:
            data = episode_data[key]
            expected_shape = lowdim_configs[key]['shape']
            if len(expected_shape)==1:
                if expected_shape[0] == 1 and len(data.shape) == 1:
                    continue

            if len(data.shape) >= 1:
                if data.shape[-len(expected_shape):] != expected_shape:
                    print(f"Warning: {key} shape mismatch. Expected: {expected_shape}, Got: {data.shape}")
                    return False
        else:
            print(f"Warning: Expected lowdim key '{key}' not found in episode data")
            return False
    
    action_shape_meta = shape_meta.get('action', {})
    if 'action' in episode_data and 'shape' in action_shape_meta:
        expected_action_shape = tuple(action_shape_meta['shape'])
        actual_action_shape = episode_data['action'].shape[-len(expected_action_shape):]
        if actual_action_shape != expected_action_shape:
            print(f"Warning: Action shape mismatch. Expected: {expected_action_shape}, Got: {actual_action_shape}")
            return False
    
    return True


def _get_replay_buffer(
        dataset_path: str,
        shape_meta: dict,
        store: Optional[zarr.ABSStore] = None,
        pc_preprocessor: Optional[PointCloudPreprocessor] = None,
        lowdim_preprocessor: Optional[LowDimPreprocessor] = None,
        downsample_factor: int = 3,
        downsample_use_all_offsets: bool = False,
        max_episodes: Optional[int] = None,
        n_workers: int = 1
) -> ReplayBuffer:

    if store is None:
        store = zarr.MemoryStore()
        
    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    False
    pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs = parse_shape_meta(shape_meta)
    
    print(f"Parsed shape_meta:")
    print(f"  - Pointcloud keys: {pointcloud_keys}")
    print(f"  - Lowdim keys: {lowdim_keys}")
    print(f"  - Action shape: {shape_meta.get('action', {}).get('shape', 'undefined')}")
    print(f"  - downsample_factor: {downsample_factor}")
    episode_dirs = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and item.name.startswith('episode_'):
            episode_dirs.append(item)
            
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
        
    print(f"Found {len(episode_dirs)} episodes to process")
    
    if len(episode_dirs) == 0:
        raise ValueError("No episode directories found")
    
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)
    
    with tqdm(total=len(episode_dirs), desc="Processing episodes", mininterval=1.0) as pbar:
        offsets = list(range(downsample_factor)) if downsample_use_all_offsets else [0]
        for episode_dir in episode_dirs:
            try:
                for off in offsets:
                    episode_data = process_single_episode(
                        episode_dir, 
                        pc_preprocessor, 
                        lowdim_preprocessor, 
                        downsample_factor,
                        downsample_offset=off
                    )

                    if episode_data is not None:
                        if validate_episode_data_with_shape_meta(episode_data, shape_meta):
                            L = len(episode_data['align_timestamp'])
                            episode_data['meta_source_episode'] = np.array([episode_dir.name]*L, dtype='S64')
                            episode_data['meta_downsample_offset'] = np.full((L,), off, dtype=np.int16)
                            for key in episode_data.keys():
                                if isinstance(episode_data[key], list):
                                    episode_data[key] = np.asarray(episode_data[key])

                            replay_buffer.add_episode(episode_data,
                                object_codecs={'pointcloud': numcodecs.Pickle()})
                            pbar.set_postfix(
                                episodes=replay_buffer.n_episodes,
                                steps=replay_buffer.n_steps
                            )
                        else:
                            print(f"Skipping episode {episode_dir.name} due to shape validation failure")
                    else:
                        print(f"Skipping empty episode: {episode_dir.name}")

            except Exception as e:
                print(f"Error processing {episode_dir.name}: {e}")
                continue
                
            pbar.update(1)
    
    print(f"Successfully processed {replay_buffer.n_episodes} episodes "
        f"with {replay_buffer.n_steps} total steps")
    
    return replay_buffer