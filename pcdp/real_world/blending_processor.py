#!/usr/bin/env python3
"""
Blending Processor
Ring buffer에서 시간 동기화된 데이터를 가져와 Femto Bolt + D405 blending 수행
"""
import numpy as np
import cv2
import open3d as o3d
from typing import Tuple, Optional
from pcdp.common.RISE_transformation import rot_trans_mat


class BlendingProcessor:
    """
    Femto Bolt + D405 blending 프로세서

    Responsibilities:
    1. Ring buffer에서 최신 데이터 가져오기
    2. 시간 동기화 (capture timestamp 기준)
    3. EE pose 읽기
    4. Blending 수행 (projection_blend)
    """

    def __init__(
        self,
        femto_camera,
        d405_camera,
        piper_interface=None,
        femto_extrinsics: Optional[np.ndarray] = None,
        ee_to_d405: Optional[np.ndarray] = None,
        d405_depth_scale: Optional[float] = None,
        # Variance-based outlier filter parameters
        enable_variance_filter: bool = True,
        variance_kernel_size: int = 5,        # 5×5 윈도우
        variance_threshold: float = 50.0,   # 분산 임계값 (mm²)
    ):
        """
        Args:
            femto_camera: SingleOrbbec 인스턴스 (None for offline processing)
            d405_camera: SingleRealSense 인스턴스 (None for offline processing)
            piper_interface: PiPER 로봇 인터페이스 (None이면 dummy pose 사용)
            femto_extrinsics: Femto camera extrinsics (4x4 matrix)
            ee_to_d405: End-effector to D405 transformation (4x4 matrix)
            d405_depth_scale: D405 depth scale (for offline processing, None for online)
            enable_variance_filter: Enable variance-based outlier filter
            variance_kernel_size: Kernel size for variance filter (default: 5)
            variance_threshold: Variance threshold in mm² (default: 100.0)
        """
        self.femto = femto_camera
        self.d405 = d405_camera
        self.piper = piper_interface

        # Variance-based outlier filter settings
        self.enable_variance_filter = enable_variance_filter
        self.variance_kernel_size = variance_kernel_size
        self.variance_threshold = variance_threshold

        # Buffers
        self.femto_buffer = None
        self.d405_buffer = None

        # Dummy pose (PiPER 없을 때)
        self.dummy_ee_pose = np.array([0.3, 0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)

        # D405 depth scale (RealSense depth unit)
        # Can be set explicitly for offline processing or initialized lazily from camera
        self.d405_depth_scale = d405_depth_scale

        # Femto camera → Base (from PCDP real_data_pc_conversion.py)
        if femto_extrinsics is None:
            self.femto_cam_to_base = np.array([
                [  0.007131,  -0.91491,    0.403594,   0.05116],
                [ -0.994138,   0.003833,   0.026560,  -0.00918],
                [ -0.025717,  -0.403641,  -0.914552,   0.50821],
                [  0.,         0.,         0.,         1.     ]
            ], dtype=np.float64)
        else:
            self.femto_cam_to_base = np.array(femto_extrinsics, dtype=np.float64)

        # Base → Femto camera (inverse)
        self.base_to_femto_cam = np.linalg.inv(self.femto_cam_to_base)

        # Base → Robot base (from single_point_cloud/core/config.py)
        self.base_to_manipulator = np.array([
            [1.0, 0.0, 0.0, -0.040],   # -40mm x
            [0.0, 1.0, 0.0, -0.290],   # -290mm y
            [0.0, 0.0, 1.0, -0.030],   # -20mm z (CORRECTED)
            [0.0, 0.0, 0.0,  1.000]
        ], dtype=np.float64)

        # End-effector → D405 (Optical frame)
        if ee_to_d405 is None:
            # Hand-eye calibration result (EndPose-based, latest calibration)
            # Calibrated on 2025-12-08
            # This is ee_to_color_frame (calibration was done with RGB image)
            self.ee_to_d405_color = np.array([
                [0.011562517413824459, -0.9045824108766296, 0.4261419600598841, -0.0630571160287582],
                [0.9999276419016558, 0.009045119479890396, -0.007930748677719701, -0.006957758124793954],
                [0.003319510814811938, 0.42620282465772, 0.9046215413650787, 0.055453185424837834],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=np.float64)

            # Use ee_to_d405_color directly (calibration already includes camera mounting)
            # D405 uses color-aligned point clouds (pc.map_to(color) in single_realsense.py)
            self.ee_to_d405 = self.ee_to_d405_color
        else:
            self.ee_to_d405 = np.array(ee_to_d405, dtype=np.float64)
            # If custom ee_to_d405 is provided, we don't have ee_to_d405_color
            # Set to None to indicate it's not available
            self.ee_to_d405_color = None

        # Pixel grid cache for depth_image_to_pointcloud (optimization)
        self._pixel_grid_cache = {}

    def get_latest_data(self):
        """
        Ring buffer에서 최신 데이터 가져오기

        Returns:
            dict: {
                'femto_pc': (N, 6) array, XYZ(mm) + RGB(0-1)
                'd405_pc': (M, 6) array, XYZ(m) + RGB(0-1)
                'ee_pose': (6,) array, [x, y, z, rx, ry, rz] in meter and radians
                'timestamps': {
                    'femto_capture': float,
                    'femto_receive': float,
                    'd405_capture': float,
                    'd405_receive': float
                }
            }
        """
        # Femto buffer (k=1: 최신 1개)
        self.femto_buffer = self.femto.get(k=1, out=self.femto_buffer)
        femto_pc = self.femto_buffer['pointcloud'][-1]
        femto_capture_ts = self.femto_buffer['camera_capture_timestamp'][-1]
        femto_receive_ts = self.femto_buffer['camera_receive_timestamp'][-1]
        femto_intrinsics = self.femto_buffer['intrinsics'][-1]
        femto_depth = self.femto_buffer['depth_image'][-1]

        # D405 buffer (k=2: 최신 2개, 시간 동기화 위해)
        self.d405_buffer = self.d405.get(k=2, out=self.d405_buffer)
        d405_pc = self.d405_buffer['pointcloud'][-1]
        d405_depth = self.d405_buffer['depth_image'][-1]
        d405_color = self.d405_buffer['color_image'][-1]
        d405_intrinsics = self.d405_buffer['intrinsics'][-1]
        d405_capture_ts = self.d405_buffer['camera_capture_timestamp'][-1]
        d405_receive_ts = self.d405_buffer['camera_receive_timestamp'][-1]

        # EE pose 읽기
        ee_pose = self._get_ee_pose()

        return {
            'femto_pc': femto_pc,
            'd405_pc': d405_pc,
            'd405_depth': d405_depth,
            'd405_color': d405_color,
            'd405_intrinsics': d405_intrinsics,
            'ee_pose': ee_pose,
            'femto_intrinsics': femto_intrinsics,
            'femto_depth': femto_depth,
            'timestamps': {
                'femto_capture': femto_capture_ts,
                'femto_receive': femto_receive_ts,
                'd405_capture': d405_capture_ts,
                'd405_receive': d405_receive_ts
            }
        }

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Apply 4x4 transformation to Nx3 points."""
        if points.shape[0] == 0:
            return points

        # Add homogeneous coordinate
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        # Transform
        transformed = (transform_matrix @ points_h.T).T
        # Return XYZ
        return transformed[:, :3]

    def _get_ee_pose(self):
        """
        PiPER에서 EE pose 읽기 (read-only)

        Returns:
            (6,) array: [x, y, z, rx, ry, rz] in meter and radians
        """
        if self.piper is None:
            return self.dummy_ee_pose.copy()

        try:
            # PiPER V2 API (new version returns list directly in mm and degrees)
            raw_pose = self.piper.GetArmEndPoseMsgs()  # [x, y, z, rx, ry, rz] in mm, deg

            ee_pose = np.array(raw_pose, dtype=np.float64)

            # New SDK returns mm and degrees directly (not 0.001mm/0.001deg)
            ee_pose[:3] *= 1e-3  # mm → m
            ee_pose[3:] = np.deg2rad(ee_pose[3:])  # deg → rad

            return ee_pose

        except Exception as e:
            print(f"[BlendingProcessor] Warning: Failed to read EE pose: {e}")
            return self.dummy_ee_pose.copy()

    def projection_blend(
        self,
        femto_pc: np.ndarray,
        d405_pc: np.ndarray,
        ee_pose: np.ndarray,
        femto_intrinsics: Tuple[float, float, float, float],
        femto_depth: np.ndarray,
        return_labeled: bool = False,
    ):
        """
        Projection-based blending with KDTree spatial deduplication (vectorized, no for-loops!)

        Strategy:
        1. Transform D405 to Femto camera frame
        2. blending logic에 의해서 융합하기
            * Femto bolt 좌표가 메인 카메라이므로 비교적 옳은 정보라고 가정 *
            1) z_d > z_f : D405와 femto bolt 좌표 둘 다 사용
            2) z_d ~=(유사) z_f: Femto bolt 좌표 사용
            3) z_d < z_f : Femto bolt 좌표 사용
        3. xyzrgb 형태로 반환

        Args:
            femto_pc: Nx6 [x,y,z,r,g,b] in Femto camera frame, mm (input), RGB 0-1
            d405_pc: Mx6 [x,y,z,r,g,b] in D405 camera frame, m, RGB 0-1
                     (Already filtered by threshold filter in SingleRealSense)
            ee_pose: [x,y,z,rx,ry,rz] end-effector pose, m, radians
            return_labeled: If True, return (N, 7) with label column (femto=0, d405=1)

        Returns:
            if return_labeled=False (default):
                blended_pc: Kx6 [x,y,z,r,g,b] in Femto camera frame, **mm**, RGB 0-1
            if return_labeled=True:
                labeled_pc: Kx7 [x,y,z,r,g,b,label] in Femto camera frame, **mm**, RGB 0-1
                            label: 0=femto, 1=d405
        """

        # 1. d405 좌표를 femto bolt depth image에 투영
        # D405 점군을 Femto bolt depth 좌표계로 변환
        manipulator_to_ee = rot_trans_mat(ee_pose[:3], ee_pose[3:6])
        base_to_ee = self.base_to_manipulator @ manipulator_to_ee
        base_to_d405 = base_to_ee @ self.ee_to_d405
        d405_to_femto = self.base_to_femto_cam @ base_to_d405

        # 단위 변환
        d405_xyz_m = d405_pc[:, :3]
        d405_xyz_femto_m = self.transform_points(d405_xyz_m, d405_to_femto)
        d405_xyz_femto_mm = d405_xyz_femto_m * 1000.0

        # D405 RGB 저장 
        d405_rgb = np.clip(d405_pc[:, 3:6], 0, 1)

        # 유효한 z 값만 남기고 데이터 추출
        z_d405 = d405_xyz_femto_mm[:, 2]
        z_valid_mask = z_d405 > 0

        xyz_valid = d405_xyz_femto_mm[z_valid_mask]
        rgb_valid = d405_rgb[z_valid_mask]

        # 만약 유효한 점이 하나도 없다면 Femto만 반환
        if len(xyz_valid) == 0:
            femto_valid = femto_pc[femto_pc[:, 2] > 0].astype(np.float32)
            if return_labeled and len(femto_valid) > 0:
                # Label column 추가: femto=0
                femto_labels = np.zeros((len(femto_valid), 1), dtype=np.float32)
                return np.hstack([femto_valid, femto_labels])
            elif return_labeled:
                return np.empty((0, 7), dtype=np.float32)
            return femto_valid

        # 투영 공식 적용
        fx, fy, cx, cy = femto_intrinsics

        u = (fx * (xyz_valid[:, 0] / xyz_valid[:, 2]) + cx).astype(np.int32)
        v = (fy * (xyz_valid[:, 1] / xyz_valid[:, 2]) + cy).astype(np.int32)

        # 이미지 경계 내 좌표 필터링
        h, w = femto_depth.shape
        in_image_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        u_valid = u[in_image_mask]
        v_valid = v[in_image_mask]

        # 비교를 위한 거리값 추출
        z_d = xyz_valid[in_image_mask, 2]
        z_f = femto_depth[v_valid, u_valid].astype(np.float32)


        # 2. blending logic 수행
        # Keep D405 points if:
        # 1. Femto has no depth (z_f == 0) - D405 sees areas Femto cannot
        # 2. D405 depth significantly differs from Femto (occlusion case)
        #    - D405 is wrist-mounted and can see behind robot arm
        occlusion_threshold = 50.0  # mm
        keep_mask = (z_f == 0) | (np.abs(z_d - z_f) > occlusion_threshold)

        # 3. D405 점들 분류
        # 3-1. Femto 시야 내의 D405 점들 (blending logic 적용)
        d405_in_view_xyz = xyz_valid[in_image_mask][keep_mask]  # mm
        d405_in_view_rgb = rgb_valid[in_image_mask][keep_mask]
        u_keep = u[in_image_mask][keep_mask]
        v_keep = v[in_image_mask][keep_mask]

        # 3-2. Femto 시야 밖의 D405 점들 (depth image에 없음, 항상 포함)
        d405_out_view_xyz = xyz_valid[~in_image_mask]  # mm
        d405_out_view_rgb = rgb_valid[~in_image_mask]

        # ========== 4. Combined Depth Image 생성 ==========
        combined_depth = femto_depth.astype(np.float32).copy()

        # D405 in-view 점들의 depth를 fusion (min depth 선택)
        if len(u_keep) > 0:
            z_keep = d405_in_view_xyz[:, 2]
            current_depth = combined_depth[v_keep, u_keep]
            update_mask = (current_depth == 0) | (z_keep < current_depth)
            combined_depth[v_keep[update_mask], u_keep[update_mask]] = z_keep[update_mask]

        # ========== 5. Variance Filter (optional) ==========
        if self.enable_variance_filter:
            filtered_depth = self.apply_variance_filter(combined_depth)
        else:
            filtered_depth = combined_depth

        # ========== 6. filtered_depth 기준 Point Cloud 필터링 ==========
        # Femto points
        femto_valid_pc = femto_pc[femto_pc[:, 2] > 0]
        femto_xyz = femto_valid_pc[:, :3]
        femto_u = (fx * femto_xyz[:, 0] / femto_xyz[:, 2] + cx).astype(np.int32)
        femto_v = (fy * femto_xyz[:, 1] / femto_xyz[:, 2] + cy).astype(np.int32)

        femto_in_bounds = (femto_u >= 0) & (femto_u < w) & (femto_v >= 0) & (femto_v < h)
        femto_keep_mask = np.zeros(len(femto_valid_pc), dtype=bool)
        femto_keep_mask[femto_in_bounds] = filtered_depth[femto_v[femto_in_bounds], femto_u[femto_in_bounds]] > 0
        femto_filtered = femto_valid_pc[femto_keep_mask]

        # D405 in-view points (filtered_depth 기준)
        if len(u_keep) > 0:
            d405_in_keep_mask = filtered_depth[v_keep, u_keep] > 0
            d405_in_view_pc = np.hstack([d405_in_view_xyz, d405_in_view_rgb])
            d405_in_filtered = d405_in_view_pc[d405_in_keep_mask]
        else:
            d405_in_filtered = np.empty((0, 6), dtype=np.float32)

        # D405 out-of-view points (depth image 밖, 항상 포함)
        if len(d405_out_view_xyz) > 0:
            d405_out_view_pc = np.hstack([d405_out_view_xyz, d405_out_view_rgb])
        else:
            d405_out_view_pc = np.empty((0, 6), dtype=np.float32)

        # ========== 7. 결과 합치기 ==========
        if return_labeled:
            # Label column 추가: femto=0, d405=1
            # femto_filtered: (N, 6) → (N, 7)
            if len(femto_filtered) > 0:
                femto_labels = np.zeros((len(femto_filtered), 1), dtype=np.float32)
                femto_labeled = np.hstack([femto_filtered, femto_labels])
            else:
                femto_labeled = np.empty((0, 7), dtype=np.float32)

            # d405_in_filtered + d405_out_view_pc → (M, 7)
            d405_parts = []
            if len(d405_in_filtered) > 0:
                d405_parts.append(d405_in_filtered)
            if len(d405_out_view_pc) > 0:
                d405_parts.append(d405_out_view_pc)

            if d405_parts:
                d405_combined = np.vstack(d405_parts)
                d405_labels = np.ones((len(d405_combined), 1), dtype=np.float32)
                d405_labeled = np.hstack([d405_combined, d405_labels])
            else:
                d405_labeled = np.empty((0, 7), dtype=np.float32)

            # 합치기
            parts = []
            if len(femto_labeled) > 0:
                parts.append(femto_labeled)
            if len(d405_labeled) > 0:
                parts.append(d405_labeled)

            result = np.vstack(parts) if parts else np.empty((0, 7), dtype=np.float32)
            return result.astype(np.float32)

        # 기존 동작: label 없이 반환
        parts = [femto_filtered]
        if len(d405_in_filtered) > 0:
            parts.append(d405_in_filtered)
        if len(d405_out_view_pc) > 0:
            parts.append(d405_out_view_pc)

        result = np.vstack(parts) if len(parts) > 0 else np.empty((0, 6), dtype=np.float32)

        return result.astype(np.float32)

    def apply_variance_filter(
        self,
        depth_image: np.ndarray,
        kernel_size: int = None,
        variance_threshold: float = None
    ) -> np.ndarray:
        """
        Depth image에 분산 기반 outlier 필터 적용 (OpenCV 가속).

        원리:
        - 각 픽셀의 주변 kernel 영역의 depth 분산 계산
        - 분산이 threshold보다 크면 해당 픽셀은 outlier로 판단
        - Var = E[X²] - E[X]² 공식 사용

        Args:
            depth_image: (H, W) depth image, mm 단위
            kernel_size: 윈도우 크기 (홀수), None이면 self.variance_kernel_size 사용
            variance_threshold: 분산 임계값 (mm²), None이면 self.variance_threshold 사용

        Returns:
            filtered_depth: (H, W) outlier 픽셀이 0으로 설정된 depth image
        """
        if kernel_size is None:
            kernel_size = self.variance_kernel_size
        if variance_threshold is None:
            variance_threshold = self.variance_threshold

        # float32로 변환 (정밀도 및 연산 속도)
        depth = depth_image.astype(np.float32)

        # 유효하지 않은 depth(0)를 처리
        valid_mask = depth > 0
        depth_zero_filled = np.where(valid_mask, depth, 0)

        # cv2.boxFilter로 local sum 계산 (normalize=False)
        ksize = (kernel_size, kernel_size)

        # 합계 계산
        depth_sum = cv2.boxFilter(depth_zero_filled, -1, ksize, normalize=False)
        valid_count = cv2.boxFilter(valid_mask.astype(np.float32), -1, ksize, normalize=False)

        # 0으로 나누기 방지 (분산 계산용)
        valid_count_safe = np.maximum(valid_count, 1)

        # E[X] = local mean
        mean = depth_sum / valid_count_safe

        # E[X²] = local mean of squared values
        depth_sq = depth_zero_filled ** 2
        depth_sq_sum = cv2.boxFilter(depth_sq, -1, ksize, normalize=False)
        sq_mean = depth_sq_sum / valid_count_safe

        # Var = E[X²] - E[X]²
        variance = sq_mean - mean ** 2

        # 분산이 threshold 이하이고 depth가 유효한 픽셀만 유지
        keep_mask = (variance < variance_threshold) & valid_mask

        # 필터링된 depth 생성
        filtered_depth = np.where(keep_mask, depth_image, 0)

        return filtered_depth.astype(depth_image.dtype)

    def apply_voxel_filter(
        self,
        point_cloud: np.ndarray,
        voxel_size: float = 0.005
    ) -> np.ndarray:
        """
        Apply voxel downsampling to point cloud using pandas hash.
        ~5x faster than np.unique (6.8ms → 1.4ms for 50k points).

        Args:
            point_cloud: (N, 6) array [x, y, z, r, g, b], XYZ in meters, RGB in 0-1
            voxel_size: Voxel size in meters (default: 0.005m = 5mm)

        Returns:
            downsampled_pc: (M, 6) array, voxel downsampled point cloud
        """
        if len(point_cloud) == 0:
            return point_cloud

        import pandas as pd

        # Quantize to voxel grid
        xyz = point_cloud[:, :3]
        voxel_coords = np.floor(xyz / voxel_size).astype(np.int32)

        # pandas 기반 unique (해시 테이블, O(n)) - np.unique보다 5배 빠름
        df = pd.DataFrame(voxel_coords, columns=['x', 'y', 'z'])
        unique_idx = df.drop_duplicates().index.values

        return point_cloud[unique_idx].astype(np.float32)

        # === 이전 방식 (np.unique, 느림 - 필요시 복원) ===
        # voxel_coords_cont = np.ascontiguousarray(voxel_coords)
        # voxel_keys = voxel_coords_cont.view(dtype=np.dtype((np.void, 12))).ravel()
        # _, unique_idx = np.unique(voxel_keys, return_index=True)
        # return point_cloud[unique_idx].astype(np.float32)

    def _get_pixel_grid(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached pixel coordinate grid for given resolution."""
        key = (h, w)
        if key not in self._pixel_grid_cache:
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            self._pixel_grid_cache[key] = (u.flatten().astype(np.float32),
                                            v.flatten().astype(np.float32))
        return self._pixel_grid_cache[key]

    def depth_image_to_pointcloud(
        self,
        depth_image: np.ndarray,
        color_image: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        depth_scale: float = 1.0
    ) -> np.ndarray:
        """
        Convert depth image + aligned color image to point cloud.
        Uses cached pixel grids for faster processing.

        Args:
            depth_image: (H, W) uint16, depth in mm (for Femto) or raw uint16 (for D405)
            color_image: (H, W, 3) uint8, RGB 0-255, aligned to depth resolution
            intrinsics: (fx, fy, cx, cy)
            depth_scale: Depth scale factor (1.0 for Femto mm, RealSense depth_scale for D405)

        Returns:
            pc: (N, 6) [x, y, z, r, g, b], XYZ in meters, RGB in 0-1
        """
        fx, fy, cx, cy = intrinsics
        h, w = depth_image.shape

        # Verify color image is aligned (same resolution as depth)
        assert color_image.shape[:2] == (h, w), \
            f"Color image {color_image.shape[:2]} must match depth resolution {(h, w)}"

        # Get cached pixel coordinates grid (major speedup)
        u, v = self._get_pixel_grid(h, w)

        # Get depth values
        z = depth_image.ravel().astype(np.float32) * depth_scale

        # Filter valid depths (z > 0)
        valid_mask = z > 0
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z[valid_mask]

        # Backproject to 3D
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy

        # Get colors at valid depth coordinates (color is aligned to depth)
        colors = color_image.reshape(-1, 3)[valid_mask].astype(np.float32) / 255.0

        # Stack XYZ + RGB
        pc = np.column_stack([x, y, z_valid, colors])

        return pc.astype(np.float32)

    def process_from_images(
        self,
        femto_color: np.ndarray,
        femto_depth: np.ndarray,
        femto_intrinsics: np.ndarray,
        d405_color: np.ndarray,
        d405_depth: np.ndarray,
        d405_intrinsics: np.ndarray,
        ee_pose: np.ndarray,
        d405_depth_scale: float = 0.0001,  # RealSense D405 actual: 0.0001m per unit
    ) -> np.ndarray:
        """
        Process images to blended point cloud (for training)

        NOTE: d405_depth already has SDK threshold + spatial filters applied during collection.
        No additional filtering is needed here.

        Args:
            femto_color: (720, 1280, 3) uint8
            femto_depth: (288, 320) uint16, in mm
            femto_intrinsics: (4,) [fx, fy, cx, cy]
            d405_color: (240, 424, 3) uint8
            d405_depth: (240, 424) uint16, SDK filtered depth values
            d405_intrinsics: (4,) [fx, fy, cx, cy]
            ee_pose: (6,) [x, y, z, rx, ry, rz] in meter and radians
            d405_depth_scale: D405 depth scale (meters per unit)
            apply_threshold_filter: Deprecated, ignored (filters applied during collection)
            threshold_min_dist: Deprecated, ignored
            threshold_max_dist: Deprecated, ignored

        Returns:
            blended_pc: (N, 6) [x, y, z, r, g, b] in Femto camera frame, **mm**, RGB 0-1
        """
        # 1. Femto Bolt: depth image to point cloud
        # Femto depth is already in mm, and we want output in mm for blending
        femto_pc = self.depth_image_to_pointcloud(
            depth_image=femto_depth,
            color_image=femto_color,
            intrinsics=tuple(femto_intrinsics),
            depth_scale=1.0  # Already in mm
        )
        # Convert from meters to mm (depth_image_to_pointcloud outputs meters)
        femto_pc[:, :3] *= 1000.0  # m -> mm

        # 2. D405: Convert depth image to point cloud
        # NOTE: d405_depth already has SDK threshold + spatial filters applied
        d405_pc = self.depth_image_to_pointcloud(
            depth_image=d405_depth,
            color_image=d405_color,
            intrinsics=tuple(d405_intrinsics),
            depth_scale=d405_depth_scale  # Convert depth values to meters
        )
        # d405_pc is now in meters

        # 4. Apply voxel downsampling to D405 point cloud (5mm)
        d405_pc_voxel = self.apply_voxel_filter(
            point_cloud=d405_pc,
            voxel_size=0.005  # 5mm
        )

        # 5. Perform blending
        blended_pc = self.projection_blend(
            femto_pc=femto_pc,
            d405_pc=d405_pc_voxel,
            ee_pose=ee_pose,
            femto_intrinsics=tuple(femto_intrinsics),
            femto_depth=femto_depth
        )

        return blended_pc

    def process(self, return_labeled: bool = False):
        """
        전체 파이프라인: 데이터 가져오기 → OpenCV 필터 적용 → Blending

        Args:
            return_labeled: If True, blended_pc includes label column (femto=0, d405=1)

        Returns:
            dict: {
                'femto_pc': (N, 6) array, raw Femto PC (camera frame, mm)
                'd405_pc': (M, 6) array, raw D405 PC (camera frame, m)
                'blended_pc': (K, 6) or (K, 7) array, blended PC (Femto frame, mm)
                              if return_labeled=True: 7th column is label (0=femto, 1=d405)
                'ee_pose': (6,) array,
                'timestamps': dict,
            }
        """
        # Lazy initialization: Get D405 depth scale on first call
        if self.d405_depth_scale is None:
            if not self.d405.is_ready:
                raise RuntimeError("D405 camera is not ready. Call d405.start() first.")
            self.d405_depth_scale = self.d405.get_depth_scale()
            print(f"[BlendingProcessor] D405 depth scale initialized: {self.d405_depth_scale}")

        # 1. 최신 데이터 가져오기 (Femto)
        data = self.get_latest_data()

        # 2. D405 depth_image → point cloud (SDK threshold + spatial filters already applied)
        d405_pc = self.depth_image_to_pointcloud(
            depth_image=data['d405_depth'],
            color_image=data['d405_color'],
            intrinsics=tuple(data['d405_intrinsics']),
            depth_scale=self.d405_depth_scale
        )

        # 3. Blending 수행
        blended_pc = self.projection_blend(
            data['femto_pc'],
            d405_pc,
            data['ee_pose'],
            data['femto_intrinsics'],
            data['femto_depth'],
            return_labeled=return_labeled
        )

        return {
            'femto_pc': data['femto_pc'],
            'd405_pc': d405_pc,
            'blended_pc': blended_pc,
            'ee_pose': data['ee_pose'],
            'timestamps': data['timestamps']
        }
