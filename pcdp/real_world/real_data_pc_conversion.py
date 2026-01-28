# real_data_pc_conversion_v2.py



from typing import Tuple, Optional, List
import pathlib
import cv2
import numpy as np
import zarr
import numcodecs
from tqdm import tqdm
from dataclasses import dataclass

from pcdp.common import RISE_transformation as rise_tf
from pcdp.common.replay_buffer import ReplayBuffer

# === Fixed sensor defaults (hardcoded) ===
# Femto depth/pointcloud: millimeters; D405 depth: 0.1mm (RealSense scale)
FEMTO_DEPTH_W, FEMTO_DEPTH_H = 320, 288
D405_STRIDE = 1

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])



def _parse_intrinsics(intr) -> Tuple[float, float, float, float]:
    """
    Supports:
      - dict: {"fx","fy","cx","cy"}
      - np.ndarray (4,) or (3,3)
      - object with fx,fy,cx,cy
    """
    if intr is None:
        raise ValueError("intrinsics is None")

    if isinstance(intr, dict):
        return float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])

    if isinstance(intr, (list, tuple)):
        intr = np.asarray(intr)

    if isinstance(intr, np.ndarray):
        if intr.shape == (4,):
            fx, fy, cx, cy = intr.tolist()
            return float(fx), float(fy), float(cx), float(cy)
        if intr.shape == (3, 3):
            fx = intr[0, 0]; fy = intr[1, 1]
            cx = intr[0, 2]; cy = intr[1, 2]
            return float(fx), float(fy), float(cx), float(cy)

    if all(hasattr(intr, k) for k in ("fx", "fy", "cx", "cy")):
        return float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)

    raise ValueError(f"Unsupported intrinsics format: type={type(intr)}")


def _femto_depth_to_meters(depth: np.ndarray) -> np.ndarray:
    return depth.astype(np.float32, copy=False) * 1e-3


def _d405_depth_to_meters(depth: np.ndarray) -> np.ndarray:
    return depth.astype(np.float32, copy=False) * 1e-4


def crop_workspace_points(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    points: (N,3) or (N,6) or (N,7) (앞 3개 xyz)
    bounds: (3,2)
    """
    if points is None or points.size == 0:
        return points
    xyz = points[:, :3]
    m = (
        (xyz[:, 0] >= bounds[0, 0]) & (xyz[:, 0] <= bounds[0, 1]) &
        (xyz[:, 1] >= bounds[1, 0]) & (xyz[:, 1] <= bounds[1, 1]) &
        (xyz[:, 2] >= bounds[2, 0]) & (xyz[:, 2] <= bounds[2, 1])
    )
    return points[m]


def transform_points(T_4x4: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if xyz is None or xyz.size == 0:
        return xyz
    xyz = np.asarray(xyz)
    N = xyz.shape[0]
    xyz_h = np.c_[xyz, np.ones((N, 1), dtype=np.float64)]
    out = (T_4x4 @ xyz_h.T).T[:, :3]
    return out


def voxel_keys_from_xyz(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    xyz: (N,3) float
    return: (N,) np.void keys
    """
    if xyz.size == 0:
        return np.empty((0,), dtype=np.dtype((np.void, 12)))
    grid = np.floor(xyz / float(voxel_size)).astype(np.int32, copy=False)
    grid = np.ascontiguousarray(grid)
    return grid.view(np.dtype((np.void, 12))).ravel()


# variance filter + depth threshold
def depth_variance_filter(
    depth: np.ndarray,
    depth_to_meters_fn,
    ksize: int = 5,
    std_thresh_m: float = 0.004,   # 4mm 정도부터 시작 추천
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    """
    depth image에서 로컬 분산(표준편차) 기반으로 noisy pixel 제거.
    - 반환은 depth와 동일 dtype 유지(대부분 uint16), 제거 픽셀은 0.
    - 계산은 meter 단위로 함.
    """
    if depth is None:
        raise ValueError("depth is None")
    if ksize % 2 == 0:
        ksize += 1

    depth_m = depth_to_meters_fn(depth)
    valid = (depth_m > float(min_depth_m)) & (depth_m < float(max_depth_m))
    if not np.any(valid):
        out = np.zeros_like(depth)
        return out

    # invalid는 0으로 두고 boxFilter 수행
    x = np.where(valid, depth_m, 0.0).astype(np.float32)
    ones = valid.astype(np.float32)

    # E[x], E[x^2] (valid만 평균)
    mean_x = cv2.boxFilter(x, ddepth=-1, ksize=(ksize, ksize), normalize=False, borderType=cv2.BORDER_DEFAULT)
    mean_x2 = cv2.boxFilter(x * x, ddepth=-1, ksize=(ksize, ksize), normalize=False, borderType=cv2.BORDER_DEFAULT)
    cnt = cv2.boxFilter(ones, ddepth=-1, ksize=(ksize, ksize), normalize=False, borderType=cv2.BORDER_DEFAULT)
    cnt = np.maximum(cnt, 1.0)

    ex = mean_x / cnt
    ex2 = mean_x2 / cnt
    var = np.maximum(ex2 - ex * ex, 0.0)
    std = np.sqrt(var)

    keep = valid & (std <= float(std_thresh_m))
    out = np.where(keep, depth, 0).astype(depth.dtype, copy=False)
    return out


def d405_depth_to_pointcloud(
    depth: np.ndarray,
    color: Optional[np.ndarray],
    intrinsics,
    stride: int = 2,
) -> np.ndarray:
    """
    depth+color -> (N,6) camera frame xyz(m) + rgb(0~1)
    """
    if depth is None:
        return np.zeros((0, 6), dtype=np.float32)

    fx, fy, cx, cy = _parse_intrinsics(intrinsics)
    # Depth scale is sensor-fixed; ignore provided unit and use defaults
    # D405 depth is 0.1mm scale (fixed)
    depth_m = _d405_depth_to_meters(depth)

    H, W = depth_m.shape
    if color is None:
        color = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        if color.shape[0] != H or color.shape[1] != W:
            H2 = min(H, color.shape[0])
            W2 = min(W, color.shape[1])
            depth_m = depth_m[:H2, :W2]
            color = color[:H2, :W2]
            H, W = H2, W2

    v = np.arange(0, H, stride, dtype=np.int32)
    u = np.arange(0, W, stride, dtype=np.int32)
    uu, vv = np.meshgrid(u, v)
    z = depth_m[vv, uu]

    # max_depth_m pruning is already applied upstream in depth_variance_filter
    valid = (z > 0.0)
    if not np.any(valid):
        return np.zeros((0, 6), dtype=np.float32)

    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (uu - float(cx)) / float(fx) * z
    y = (vv - float(cy)) / float(fy) * z
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    rgb = color[vv.astype(np.int32), uu.astype(np.int32), :3].astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    return np.concatenate([xyz, rgb], axis=1).astype(np.float32, copy=False)


def femto_filter_pointcloud_by_depth_mask(
    femto_points_cam: np.ndarray,     # (N,6) xyz + rgb, Femto camera frame (mm)
    depth_masked: np.ndarray,         # variance-filter 적용된 depth (0이면 제거)
    K: np.ndarray,                    # 3x3 intrinsics
    z_consistency_eps_m: float = 0.01,
    use_z_consistency: bool = True,
) -> np.ndarray:
    """
    Femto처럼 depth로 PC 재생성이 불가한 경우:
    - variance-filter된 depth에서 살아남은 픽셀만 유지하도록
      point cloud를 다시 투영(u,v)해서 매칭 후 제거.
    """
    if femto_points_cam is None or femto_points_cam.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if depth_masked is None:
        return femto_points_cam.astype(np.float32, copy=False)

    H, W = depth_masked.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    xyz = femto_points_cam[:, :3].astype(np.float32, copy=False)

    # femto pointcloud는 mm 단위로 가정
    xyz_m = xyz * 1e-3

    z = xyz_m[:, 2]
    valid_z = z > 1e-6
    if not np.any(valid_z):
        return np.zeros((0, 6), dtype=np.float32)

    x = xyz_m[:, 0]
    y = xyz_m[:, 1]
    u = np.rint(fx * (x / z) + cx).astype(np.int32)
    v = np.rint(fy * (y / z) + cy).astype(np.int32)

    inb = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(inb):
        return np.zeros((0, 6), dtype=np.float32)

    idx_inb = np.flatnonzero(inb)
    u_in = u[inb]
    v_in = v[inb]

    # depth_masked에서 0이 아닌 픽셀만 keep
    ok_pix = depth_masked[v_in, u_in] > 0
    if not np.any(ok_pix):
        return np.zeros((0, 6), dtype=np.float32)

    keep_mask = np.zeros((femto_points_cam.shape[0],), dtype=bool)

    if use_z_consistency:
        depth_m = _femto_depth_to_meters(depth_masked)
        z_pc = z[inb][ok_pix]
        z_img = depth_m[v_in[ok_pix], u_in[ok_pix]]
        ok_z = np.abs(z_pc - z_img) <= float(z_consistency_eps_m)
        keep_mask[idx_inb[ok_pix][ok_z]] = True
    else:
        keep_mask[idx_inb[ok_pix]] = True

    return femto_points_cam[keep_mask].astype(np.float32, copy=False)


def _project_points_to_pixels(points_cam_xyz_m: np.ndarray, K: np.ndarray, width: int, height: int):
    """
    points_cam_xyz_m: (N,3) meters, camera frame
    return:
      u,v,z  (all filtered to in-bounds and z>0)
    """
    if points_cam_xyz_m is None or points_cam_xyz_m.size == 0:
        return (np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32))

    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    x = points_cam_xyz_m[:, 0].astype(np.float32, copy=False)
    y = points_cam_xyz_m[:, 1].astype(np.float32, copy=False)
    z = points_cam_xyz_m[:, 2].astype(np.float32, copy=False)

    valid = z > 1e-6
    if not np.any(valid):
        return (np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32))

    u = np.rint(fx * (x[valid] / z[valid]) + cx).astype(np.int32)
    v = np.rint(fy * (y[valid] / z[valid]) + cy).astype(np.int32)
    z = z[valid]

    inb = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inb], v[inb], z[inb]


@dataclass
class D405Config:
    d405_cam_to_eef: np.ndarray                 # (4,4) eef<-cam
    robot_to_base: np.ndarray                   # (4,4) base<-robot  (너 코드의 ROBOT_TO_BASE)
    stride: int = D405_STRIDE
    max_depth_m: float = 0.25                   # can be overridden from YAML
    var_ksize: int = 5
    var_std_thresh_m: float = 0.004
    var_min_depth_m: float = 0.01
    var_max_depth_m: float = 0.25


@dataclass
class FemtoConfig:
    K: np.ndarray                               # (3,3)
    cam_to_base: np.ndarray                     # (4,4) base<-cam
    width: int = FEMTO_DEPTH_W
    height: int = FEMTO_DEPTH_H
    var_ksize: int = 5
    var_std_thresh_m: float = 0.004
    var_min_depth_m: float = 0.01
    var_max_depth_m: float = 1e6


class PointCloudPreprocessor:
    """
    목표 파이프라인(요약):
      - Femto:
          femto_depth(raw)            -> (occlusion prune용 z-buffer 비교에 사용, depth==0은 unknown)
          femto_depth_var(variance)   -> femto_points를 재투영해 마스킹 -> current cloud(merge용)
      - D405:
          d405_depth(raw)             -> (occlusion prune용 z-buffer 비교에 사용, depth==0 unknown)
          d405_depth_var(variance)    -> (depth_var + color)로 PC 생성 -> current cloud(merge용)

    메모리 업데이트:
      memory(t) = prune(memory(t-1), current_raw_depths)  # free-space carve (raw depth 기반)
      memory(t) = voxel_merge(memory(t), current_points)  # current가 voxel 우선
      confidence c는 memory에만 decay, current는 c=1
    """

    def __init__(
        self,
        workspace_bounds: np.ndarray,
        femto: Optional[FemtoConfig] = None,
        d405: Optional[D405Config] = None,
        sensor_mode: str = "both",                # "femto"|"d405"|"both"
        enable_temporal: bool = True,
        enable_TF: bool = True,                   # cam->base transform
        enable_crop: bool = True,                 # workspace crop
        enable_filter: bool = True,               # variance filter
        temporal_voxel_size: float = 0.005,
        temporal_decay: float = 0.95,
        c_min: float = 0.1,
        free_space_margin_m: float = 0.009,
        verbose: bool = False,
    ):
        self.workspace_bounds = workspace_bounds.astype(np.float64)
        self.femto = femto
        self.d405 = d405
        assert sensor_mode in ("femto", "d405", "both")
        self.sensor_mode = sensor_mode
        self.enable_TF = bool(enable_TF)
        self.enable_crop = bool(enable_crop)
        self.enable_filter = bool(enable_filter)

        self.enable_temporal = bool(enable_temporal)
        self.temporal_voxel_size = float(temporal_voxel_size)
        self.temporal_decay = float(temporal_decay)
        self.c_min = float(c_min)
        self.free_space_margin_m = float(free_space_margin_m)
        self.verbose = bool(verbose)

        # memory: (M,7) xyz rgb c  in BASE frame
        self._memory = np.zeros((0, 7), dtype=np.float32)

        # cached inverses
        self._base_to_femto = None
        if self.femto is not None:
            self._base_to_femto = np.linalg.inv(self.femto.cam_to_base).astype(np.float64)

    def reset_temporal(self):
        self._memory = np.zeros((0, 7), dtype=np.float32)

    # --- (optional) old-style process for a single femto pointcloud ---
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        기존 코드 호환용: femto point cloud만 base로 변환 + workspace crop.
        (요구사항: point cloud filter는 사용 안 함)
        """
        if self.femto is None:
            raise RuntimeError("FemtoConfig is required for __call__().")
        if points is None or points.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        pts = points.astype(np.float32, copy=False)
        xyz = pts[:, :3].astype(np.float64)

        # femto pointcloud는 mm 단위로 가정
        xyz_m = xyz * 1e-3

        xyz_base = transform_points(self.femto.cam_to_base, xyz_m) if self.enable_TF else xyz_m
        rgb = pts[:, 3:6].astype(np.float32, copy=False)
        out = np.concatenate([xyz_base.astype(np.float32), rgb], axis=1)
        if self.enable_crop and self.enable_TF:
            out = crop_workspace_points(out, self.workspace_bounds)
        return out.astype(np.float32, copy=False)

    # ----------------------------
    # Helpers to build current clouds
    # ----------------------------
    def _build_femto_current(self, femto_points, femto_depth):
        if self.femto is None:
            raise RuntimeError("FemtoConfig is required for femto mode.")
        if femto_points is None or femto_depth is None:
            raise ValueError("femto_points and femto_depth are required in femto/both mode.")

        if self.enable_filter:
            femto_depth_var = depth_variance_filter(
                femto_depth,
                depth_to_meters_fn=_femto_depth_to_meters,
                ksize=self.femto.var_ksize,
                std_thresh_m=self.femto.var_std_thresh_m,
                min_depth_m=self.femto.var_min_depth_m,
                max_depth_m=self.femto.var_max_depth_m,
            )
            femto_points_masked = femto_filter_pointcloud_by_depth_mask(
                femto_points_cam=femto_points,
                depth_masked=femto_depth_var,
                K=self.femto.K,
                z_consistency_eps_m=0.02,
                use_z_consistency=True,
            )
        else:
            femto_depth_var = femto_depth
            femto_points_masked = femto_points

        femto_base = self.__call__(femto_points_masked)
        view = (femto_depth, _femto_depth_to_meters, self._base_to_femto, self.femto.K, (self.femto.width, self.femto.height))
        return femto_base, view

    def _build_d405_current(
        self,
        d405_depth,
        d405_color,
        d405_intrinsics,
        robot_eef_pose,
    ):
        if self.d405 is None:
            raise RuntimeError("D405Config is required for d405 mode.")

        if d405_depth is None or d405_intrinsics is None or robot_eef_pose is None:
            raise ValueError("d405_depth, d405_intrinsics, robot_eef_pose are required.")

        if self.enable_filter:
            d405_depth_var = depth_variance_filter(
                d405_depth,
                depth_to_meters_fn=_d405_depth_to_meters,
                ksize=self.d405.var_ksize,
                std_thresh_m=self.d405.var_std_thresh_m,
                min_depth_m=self.d405.var_min_depth_m,
                max_depth_m=self.d405.var_max_depth_m,
            )
        else:
            d405_depth_var = d405_depth

        pc_d405_cam = d405_depth_to_pointcloud(
            depth=d405_depth_var,
            color=d405_color,
            intrinsics=d405_intrinsics,
            stride=self.d405.stride,
        )  # (Nd,6) cam frame meters

        trans = robot_eef_pose[:3]
        rpy = robot_eef_pose[3:6]
        T_robot_eef = rise_tf.rot_trans_mat(trans, rpy)              # robot<-eef
        T_base_eef = self.d405.robot_to_base @ T_robot_eef           # base<-eef
        d405_cam_to_base = (T_base_eef @ self.d405.d405_cam_to_eef).astype(np.float64)  # base<-cam

        if pc_d405_cam.size > 0:
            xyz_base = transform_points(d405_cam_to_base, pc_d405_cam[:, :3].astype(np.float64)) if self.enable_TF else pc_d405_cam[:, :3].astype(np.float64)
            d405_base = np.concatenate([xyz_base.astype(np.float32), pc_d405_cam[:, 3:6].astype(np.float32)], axis=1)
            if self.enable_crop and self.enable_TF:
                d405_base = crop_workspace_points(d405_base, self.workspace_bounds)
        else:
            d405_base = np.zeros((0, 6), dtype=np.float32)

        view = None
        if d405_depth is not None and d405_cam_to_base is not None and d405_intrinsics is not None:
            fx, fy, cx, cy = _parse_intrinsics(d405_intrinsics)
            K_d = np.array([[fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
            base_to_d405 = np.linalg.inv(d405_cam_to_base).astype(np.float64)
            H, W = d405_depth.shape[:2]
            view = (d405_depth, _d405_depth_to_meters, base_to_d405, K_d, (W, H))

        return d405_base, view

    # ----------------------------
    # Main entry: process_global
    # ----------------------------
    def process_global(
        self,
        sensor_mode: Optional[str] = None,             # override: "femto"|"d405"|"both"
        # Femto inputs
        femto_points: Optional[np.ndarray] = None,      # (N,6) in Femto camera frame
        femto_depth: Optional[np.ndarray] = None,       # (H,W) raw depth image

        # D405 inputs (option A: raw depth + color + intr + pose)
        d405_depth: Optional[np.ndarray] = None,
        d405_color: Optional[np.ndarray] = None,
        d405_intrinsics=None,
        robot_eef_pose: Optional[np.ndarray] = None,    # (6,) xyz(m)+rpy(rad) in ROBOT frame

        export_mode: str = "fused",                     # "fused" only for now
    ) -> np.ndarray:
        """
        returns: fused cloud (M,7) in BASE frame: xyz rgb c
        """
        if export_mode != "fused":
            raise ValueError("This refactor currently supports export_mode='fused' only.")

        sensor_mode_local = sensor_mode or self.sensor_mode

        # ---- build current clouds (merge용) ----
        cur_list = []
        femto_view = None
        if sensor_mode_local in ("femto", "both"):
            femto_base, femto_view = self._build_femto_current(femto_points, femto_depth)
            if femto_base.size > 0:
                cur_list.append(femto_base)

        d405_view = None
        if sensor_mode_local in ("d405", "both"):
            d405_base, d405_view = self._build_d405_current(
                d405_depth=d405_depth,
                d405_color=d405_color,
                d405_intrinsics=d405_intrinsics,
                robot_eef_pose=robot_eef_pose,
            )
            if d405_base.size > 0:
                cur_list.append(d405_base)

        # current concat
        if len(cur_list) == 0:
            current = np.zeros((0, 6), dtype=np.float32)
        elif len(cur_list) == 1:
            current = cur_list[0]
        else:
            current = np.concatenate(cur_list, axis=0).astype(np.float32, copy=False)

        # ---- temporal update ----
        if not self.enable_temporal:
            # confidence 없이 fused 반환(visual 호환 위해 c=1)
            if current.size == 0:
                return np.zeros((0, 7), dtype=np.float32)
            c = np.ones((current.shape[0], 1), dtype=np.float32)
            return np.concatenate([current, c], axis=1).astype(np.float32)

        # 1) decay memory confidence + drop low c
        mem = self._memory
        if mem.size > 0:
            mem = mem.copy()
            mem[:, 6] *= self.temporal_decay
            mem = mem[mem[:, 6] >= self.c_min]

        # 2) prune old memory using raw depth views (depth==0 => unknown => keep)
        views = []
        if femto_view is not None:
            views.append(("femto",) + femto_view)
        if d405_view is not None:
            views.append(("d405",) + d405_view)

        if mem.size > 0 and len(views) > 0:
            mem = self._occlusion_prune_memory_with_raw_depth(mem, views, margin_m=self.free_space_margin_m)

        # 3) merge current into memory (voxel unique, current wins)
        fused = self._voxel_merge_memory_and_current(mem, current)

        self._memory = fused
        return fused

    # ----------------------------
    # Internals
    # ----------------------------

    def _occlusion_prune_memory_with_raw_depth(
        self,
        memory_xyzrgbc: np.ndarray,  # (M,7) in BASE
        views,                       # list of ("name", depth_raw, depth_to_m_fn, base_to_cam, K, (W,H))
        margin_m: float,
    ) -> np.ndarray:
        """
        Free-space carving rule (conservative, raw depth 기반):
          - memory point를 camera로 투영해서 (u,v,z_mem) 계산
          - current depth z_cur가 0이면 unknown -> prune 근거 없음 -> keep
          - z_cur > z_mem + margin 이면 그 ray는 memory point까지 free-space로 관측됨 -> delete
          - else keep
        Multi-view 결합: "any view에서 delete 조건 만족하면 delete"
        """
        if memory_xyzrgbc.size == 0:
            return memory_xyzrgbc

        xyz_base = memory_xyzrgbc[:, :3].astype(np.float64, copy=False)
        keep = np.ones((xyz_base.shape[0],), dtype=bool)

        for (_, depth_raw, depth_to_m_fn, base_to_cam, K, (W, H)) in views:
            if not np.any(keep):
                break

            idx = np.flatnonzero(keep)
            xyz_cam = transform_points(base_to_cam, xyz_base[idx])  # meters
            u, v, z_mem = _project_points_to_pixels(xyz_cam, K, W, H)
            if u.size == 0:
                continue

            # z_mem은 idx subset에서 valid만 남은 상태라서, 다시 매칭이 필요
            # 간단하게: valid projection mask를 다시 만들어 index mapping 하는 방식
            # -> 구현을 깔끔하게 하려면 한 번 더 전체를 projection해서 mask 얻는게 안정적이지만,
            #    여기서는 idx subset 기준으로 진행.

            # subset 기준으로 다시 projection mask를 얻기 위해 동일 계산을 재수행
            fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
            x = xyz_cam[:, 0].astype(np.float32, copy=False)
            y = xyz_cam[:, 1].astype(np.float32, copy=False)
            z = xyz_cam[:, 2].astype(np.float32, copy=False)

            valid = z > 1e-6
            if not np.any(valid):
                continue

            uu = np.rint(fx * (x[valid] / z[valid]) + cx).astype(np.int32)
            vv = np.rint(fy * (y[valid] / z[valid]) + cy).astype(np.int32)
            zz = z[valid]

            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not np.any(inb):
                continue

            uu = uu[inb]; vv = vv[inb]; zz = zz[inb]
            idx_valid = idx[np.flatnonzero(valid)][inb]  # memory index (global)

            z_cur = depth_to_m_fn(depth_raw)[vv, uu]
            known = z_cur > 0.0

            # delete 조건: z_cur > z_mem + margin
            delete = known & (z_cur > (zz + float(margin_m)))
            if np.any(delete):
                keep[idx_valid[delete]] = False

        return memory_xyzrgbc[keep]

    def _voxel_merge_memory_and_current(self, mem_xyzrgbc: np.ndarray, cur_xyzrgb: np.ndarray) -> np.ndarray:
        """
        mem: (M,7) base
        cur: (N,6) base
        output: (K,7) base
        - voxel 단위로 unique
        - current가 voxel 우선(override)
        """
        if cur_xyzrgb.size == 0 and mem_xyzrgbc.size == 0:
            return np.zeros((0, 7), dtype=np.float32)

        if cur_xyzrgb.size == 0:
            return mem_xyzrgbc.astype(np.float32, copy=False)

        cur = cur_xyzrgb.astype(np.float32, copy=False)
        cur_c = np.ones((cur.shape[0], 1), dtype=np.float32)
        cur7 = np.concatenate([cur, cur_c], axis=1)

        if mem_xyzrgbc.size == 0:
            # voxel unique만 한번
            keys = voxel_keys_from_xyz(cur7[:, :3].astype(np.float64), self.temporal_voxel_size)
            rev = np.arange(keys.shape[0]-1, -1, -1)
            _, first = np.unique(keys[rev], return_index=True)
            keep = np.sort(rev[first])
            return cur7[keep]

        # decay/prune된 mem과 current 결합 -> voxel unique (current wins)
        combined = np.concatenate([mem_xyzrgbc.astype(np.float32, copy=False), cur7], axis=0)
        keys = voxel_keys_from_xyz(combined[:, :3].astype(np.float64), self.temporal_voxel_size)

        # "last wins"를 위해 reverse unique
        rev = np.arange(keys.shape[0]-1, -1, -1)
        _, first = np.unique(keys[rev], return_index=True)
        keep = np.sort(rev[first])

        return combined[keep].astype(np.float32, copy=False)


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
