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
        enable_variance_filter: bool = False,
        variance_kernel_size: int = 5,        # 5×5 윈도우
        variance_threshold: float = 50.0,   # 분산 임계값 (mm²)
        # Composite filter parameters (Gradient + Density)
        enable_composite_filter: bool = False,
        composite_gradient_threshold: float = 80.0,  # mm (급경사 임계값)
        composite_neighbor_kernel: int = 5,          # 이웃 확인 영역 (5x5)
        composite_min_neighbors: int = 12,           # 최소 이웃 개수
        composite_enable_erosion: bool = False,       # 침식 연산 활성화
        composite_erosion_size: int = 3              # 침식 커널 크기
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

        # Composite filter settings (Gradient + Density + Erosion)
        self.enable_composite_filter = enable_composite_filter
        self.composite_gradient_threshold = composite_gradient_threshold
        self.composite_neighbor_kernel = composite_neighbor_kernel
        self.composite_min_neighbors = composite_min_neighbors
        self.composite_enable_erosion = composite_enable_erosion
        self.composite_erosion_size = composite_erosion_size

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
        return_debug: bool = False
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

        Returns:
            blended_pc: Kx6 [x,y,z,r,g,b] in Femto camera frame, **mm** (output), RGB 0-1
        """
        import time
        t_start = time.perf_counter()

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
            return femto_pc[femto_pc[:, 2] > 0].astype(np.float32)

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

        # 3. 결과 반환 (단위: mm 유지 - PointCloudPreprocessor가 mm→m 변환)
        # 3-1. Femto 시야 내의 D405 점들 (blending logic 적용)
        d405_in_view_xyz = xyz_valid[in_image_mask][keep_mask]  # mm
        d405_in_view_rgb = rgb_valid[in_image_mask][keep_mask]

        # 3-2. Femto 시야 밖의 D405 점들 (그대로 추가 - Femto가 볼 수 없는 영역)
        d405_out_view_xyz = xyz_valid[~in_image_mask]  # mm
        d405_out_view_rgb = rgb_valid[~in_image_mask]

        # 합치기
        d405_keep_xyz = np.vstack([d405_in_view_xyz, d405_out_view_xyz]) if len(d405_out_view_xyz) > 0 else d405_in_view_xyz
        d405_keep_rgb = np.vstack([d405_in_view_rgb, d405_out_view_rgb]) if len(d405_out_view_rgb) > 0 else d405_in_view_rgb

        d405_result_pc = np.hstack([d405_keep_xyz, d405_keep_rgb])
        femto_valid_pc = femto_pc[femto_pc[:, 2] > 0]
        # femto_valid_pc is already in mm

        result = np.vstack([femto_valid_pc, d405_result_pc])

        # Depth 기반 필터 적용 (Variance Filter 또는 Composite Filter)
        # 둘 중 하나만 활성화되어야 함 
        filter_enabled = self.enable_composite_filter or self.enable_variance_filter

        if filter_enabled:
            # Combined depth image 생성 (Femto + D405)
            combined_depth = femto_depth.astype(np.float32).copy()

            # D405 점들 중 keep된 점들의 depth를 combined_depth에 추가 (vectorized)
            u_keep = u[in_image_mask][keep_mask]
            v_keep = v[in_image_mask][keep_mask]
            z_keep = d405_in_view_xyz[:, 2]  # mm (in-view 점들만)

            # Vectorized depth fusion: min depth 선택 (가까운 물체 우선)
            if len(u_keep) > 0:
                # 현재 depth와 D405 depth 비교
                current_depth = combined_depth[v_keep, u_keep]
                # Femto가 0이거나 D405가 더 가까우면 D405 사용
                update_mask = (current_depth == 0) | (z_keep < current_depth)
                combined_depth[v_keep[update_mask], u_keep[update_mask]] = z_keep[update_mask]

            # 필터 선택 (Composite 우선, 아니면 Variance)
            if self.enable_composite_filter:
                # Composite filter 적용 (Gradient + Neighbor Count + Erosion)
                filtered_depth = self.apply_composite_filter(combined_depth)
            else:
                # Variance filter 적용 (OpenCV 가속, ~1ms)
                filtered_depth = self.apply_variance_filter(combined_depth)

            # 필터링된 depth image 기준으로 point cloud 재구성 (vectorized)
            # Femto: 살아남은 픽셀만
            femto_xyz = femto_valid_pc[:, :3]
            femto_z = femto_xyz[:, 2]
            femto_u = (fx * femto_xyz[:, 0] / femto_z + cx).astype(np.int32)
            femto_v = (fy * femto_xyz[:, 1] / femto_z + cy).astype(np.int32)

            # 이미지 경계 내 & filtered_depth > 0 체크 (vectorized)
            femto_in_bounds = (femto_u >= 0) & (femto_u < w) & (femto_v >= 0) & (femto_v < h)
            femto_keep_mask = np.zeros(len(femto_valid_pc), dtype=bool)
            femto_keep_mask[femto_in_bounds] = filtered_depth[femto_v[femto_in_bounds], femto_u[femto_in_bounds]] > 0

            # D405: 필터링된 depth에서 살아남은 in-view 점만 (out-of-view는 depth image에 없음)
            d405_keep_mask_final = filtered_depth[v_keep, u_keep] > 0
            d405_in_view_pc = np.hstack([d405_in_view_xyz, d405_in_view_rgb])
            d405_filtered = d405_in_view_pc[d405_keep_mask_final]

            femto_filtered = femto_valid_pc[femto_keep_mask]
            result = np.vstack([femto_filtered, d405_filtered]) if len(d405_filtered) > 0 else femto_filtered

        t_end = time.perf_counter()
        blend_time_ms = (t_end - t_start) * 1000
        # Verbose logging disabled - enable only for debugging
        # print(f"[Blending] Time: {blend_time_ms:.1f}ms | Femto: {len(femto_valid_pc)} pts | D405 in: {len(d405_pc)} pts | D405 kept: {len(d405_result_pc)} pts | Total: {len(result)} pts")

        if return_debug:
            debug_info = {
                'n_femto_valid': len(femto_valid_pc),
                'n_d405_input': len(d405_pc),
                'n_d405_kept': len(d405_result_pc),
                'n_before_filter': len(femto_valid_pc) + len(d405_result_pc),
                'n_after_filter': len(result),
                'filter_enabled': filter_enabled,
                'variance_filter_enabled': self.enable_variance_filter,
                'composite_filter_enabled': self.enable_composite_filter,
                't_blend_ms': blend_time_ms,
                't_variance_filter_ms': getattr(self, '_last_variance_filter_time_ms', 0.0),
                't_composite_filter_ms': getattr(self, '_last_composite_filter_time_ms', 0.0)
            }
            return result.astype(np.float32), debug_info

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
        import time
        t_start = time.perf_counter()

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

        t_end = time.perf_counter()
        self._last_variance_filter_time_ms = (t_end - t_start) * 1000

        return filtered_depth.astype(depth_image.dtype)

    def apply_composite_filter(
        self,
        depth_image: np.ndarray,
        gradient_threshold: float = None,
        neighbor_kernel_size: int = None,
        min_neighbors: int = None,
        enable_erosion: bool = None,
        erosion_size: int = None
    ) -> np.ndarray:
        """
        Composite filter: Gradient + Neighbor Count + Optional Erosion (OpenCV 가속).

        Flying pixel 제거를 위한 복합 필터:
        1. Gradient Filter: Sobel 연산으로 급격한 depth 변화 감지 → flying pixel 분리
        2. Neighbor Count Filter: 2D 버전의 ROR, 유효 이웃 픽셀 수로 고립된 점 제거
        3. Erosion (optional): 경계 노이즈 제거

        원리:
        - Flying pixel은 물체와 배경 사이에서 depth가 섞여 생기는 현상
        - Gradient filter로 급격한 depth 변화를 감지하여 "잘라냄"
        - 잘라낸 후 고립된 점들을 neighbor count로 제거
        - Erosion으로 경계의 남은 노이즈 정리

        Args:
            depth_image: (H, W) depth image, mm 단위
            gradient_threshold: Sobel gradient 임계값 (mm), None이면 self.composite_gradient_threshold 사용
            neighbor_kernel_size: 이웃 확인 커널 크기, None이면 self.composite_neighbor_kernel 사용
            min_neighbors: 최소 유효 이웃 개수, None이면 self.composite_min_neighbors 사용
            enable_erosion: 침식 연산 활성화, None이면 self.composite_enable_erosion 사용
            erosion_size: 침식 커널 크기, None이면 self.composite_erosion_size 사용

        Returns:
            filtered_depth: (H, W) outlier 픽셀이 0으로 설정된 depth image
        """
        import time
        t_start = time.perf_counter()

        # 파라미터 기본값 설정
        if gradient_threshold is None:
            gradient_threshold = self.composite_gradient_threshold
        if neighbor_kernel_size is None:
            neighbor_kernel_size = self.composite_neighbor_kernel
        if min_neighbors is None:
            min_neighbors = self.composite_min_neighbors
        if enable_erosion is None:
            enable_erosion = self.composite_enable_erosion
        if erosion_size is None:
            erosion_size = self.composite_erosion_size

        # float32로 변환
        depth = depth_image.astype(np.float32)
        valid_mask = depth > 0

        # ============================================
        # Step 1: Gradient Filter (급격한 depth 변화 감지)
        # ============================================
        # Sobel operator로 x, y 방향 gradient 계산
        # 0인 픽셀은 gradient 계산에서 제외하기 위해 마스킹
        depth_masked = np.where(valid_mask, depth, 0)

        grad_x = cv2.Sobel(depth_masked, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_masked, cv2.CV_32F, 0, 1, ksize=3)

        # 제곱된 값으로 비교 (속도 최적화)
        gradient_sq = grad_x**2 + grad_y**2
        threshold_sq = gradient_threshold ** 2

        # 급격한 gradient를 가진 픽셀 제거 (flying pixel 후보)
        gradient_ok = gradient_sq < threshold_sq

        # ============================================
        # Step 2: Neighbor Count Filter 
        # ============================================
        # gradient filter 적용 후의 유효 마스크
        valid_after_gradient = valid_mask & gradient_ok
        valid_after_gradient_float = valid_after_gradient.astype(np.float32)

        # 이웃 유효 픽셀 수 계산
        ksize = (neighbor_kernel_size, neighbor_kernel_size)
        neighbor_count = cv2.boxFilter(valid_after_gradient_float, -1, ksize, normalize=False)

        # 자기 자신 제외 (중앙 픽셀)
        neighbor_count = neighbor_count - valid_after_gradient_float

        # 최소 이웃 개수 이상인 픽셀만 유지
        neighbor_ok = neighbor_count >= min_neighbors

        # ============================================
        # Step 3: Erosion (optional, 경계 노이즈 제거)
        # ============================================
        keep_mask = valid_mask & gradient_ok & neighbor_ok

        if enable_erosion and erosion_size > 1:
            # 침식 연산으로 경계 픽셀 제거
            erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
            )
            keep_mask_uint8 = keep_mask.astype(np.uint8)
            keep_mask_eroded = cv2.erode(keep_mask_uint8, erosion_kernel)
            keep_mask = keep_mask_eroded.astype(bool)

        # 필터링된 depth 생성
        filtered_depth = np.where(keep_mask, depth_image, 0)

        t_end = time.perf_counter()
        self._last_composite_filter_time_ms = (t_end - t_start) * 1000

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

    def process(self, verbose: bool = False):
        """
        전체 파이프라인: 데이터 가져오기 → OpenCV 필터 적용 → Blending

        Args:
            verbose: 디버그 정보 포함 여부

        Returns:
            dict: {
                'femto_pc': (N, 6) array,
                'd405_pc': (M, 6) array,
                'blended_pc': (K, 6) array,
                'ee_pose': (6,) array,
                'timestamps': dict,
                'debug': dict (verbose=True일 때만)
            }
        """
        import time

        # Lazy initialization: Get D405 depth scale on first call
        if self.d405_depth_scale is None:
            if not self.d405.is_ready:
                raise RuntimeError("D405 camera is not ready. Call d405.start() first.")
            self.d405_depth_scale = self.d405.get_depth_scale()
            print(f"[BlendingProcessor] D405 depth scale initialized: {self.d405_depth_scale}")

        # 1. 최신 데이터 가져오기 (Femto)
        t_femto_start = time.perf_counter()
        data = self.get_latest_data()
        t_femto_end = time.perf_counter()

        # 2. D405 depth_image → point cloud (SDK threshold + spatial filters already applied)
        t_d405_start = time.perf_counter()
        d405_pc = self.depth_image_to_pointcloud(
            depth_image=data['d405_depth'],
            color_image=data['d405_color'],
            intrinsics=tuple(data['d405_intrinsics']),
            depth_scale=self.d405_depth_scale
        )
        t_d405_end = time.perf_counter()

        # 3. Apply voxel downsampling to D405 point cloud (4mm)
        t_voxel_start = time.perf_counter()
        # d405_pc_voxel = self.apply_voxel_filter(
        #     point_cloud=d405_pc,
        #     voxel_size=0.0015  # 4mm
        # )
        t_voxel_end = time.perf_counter()

        # 4. Blending 수행
        t_blend_start = time.perf_counter()
        if verbose:
            blended_pc, blend_debug = self.projection_blend(
                data['femto_pc'],
                d405_pc,
                data['ee_pose'],
                data['femto_intrinsics'],
                data['femto_depth'],
                return_debug=True
            )
        else:
            blended_pc = self.projection_blend(
                data['femto_pc'],
                d405_pc,
                data['ee_pose'],
                data['femto_intrinsics'],
                data['femto_depth']
            )
            blend_debug = {}
        t_blend_end = time.perf_counter()

        result = {
            'femto_pc': data['femto_pc'],
            'd405_pc': d405_pc,  # Voxel downsampled D405
            'blended_pc': blended_pc,
            'ee_pose': data['ee_pose'],
            'timestamps': data['timestamps']
        }

        if verbose:
            result['debug'] = {
                # Point counts
                'n_femto_raw': len(data['femto_pc']),
                'n_d405_raw': len(d405_pc),
                'n_d405_voxel': len(d405_pc),
                'n_blended': blend_debug.get('n_before_filter', len(blended_pc)),  # 필터 전
                'n_after_filter': blend_debug.get('n_after_filter', len(blended_pc)),  # 필터 후
                # Timing (ms)
                't_femto_ms': (t_femto_end - t_femto_start) * 1000,
                't_d405_ms': (t_d405_end - t_d405_start) * 1000,
                't_voxel_ms': (t_voxel_end - t_voxel_start) * 1000,
                't_blend_ms': (t_blend_end - t_blend_start) * 1000,
                't_variance_ms': blend_debug.get('t_variance_filter_ms', 0.0),
                't_composite_ms': blend_debug.get('t_composite_filter_ms', 0.0),
                # Flags
                'variance_filter_enabled': self.enable_variance_filter,
                'composite_filter_enabled': self.enable_composite_filter,
            }

        return result
