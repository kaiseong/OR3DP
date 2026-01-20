#!/usr/bin/env python3
"""
Point Cloud Visualization - 3 Windows
Femto Bolt, D405, Blending 동시 시각화
"""
import sys
import time
import argparse
import numpy as np
import open3d as o3d
from multiprocessing.managers import SharedMemoryManager

# PCDP imports
from pcdp.real_world.single_orbbec import SingleOrbbec
from pcdp.real_world.single_realsense import SingleRealSense
from pcdp.real_world.blending_processor import BlendingProcessor
from pcdp.real_world.pc_visualizer import PointCloudVisualizer

# PiPER SDK (optional)
try:
    from piper_sdk import C_PiperInterface_V2
    PIPER_AVAILABLE = True
except ImportError:
    print("Warning: PiPER SDK not available. Using dummy pose.")
    PIPER_AVAILABLE = False


def main(
    enable_femto_vis=True,
    enable_d405_vis=True,
    enable_blending_vis=True,
    verbose=False
):
    """
    Args:
        enable_femto_vis: Femto Bolt 시각화 활성화
        enable_d405_vis: D405 시각화 활성화
        enable_blending_vis: Blending 시각화 활성화
        verbose: 디버그 출력 활성화 (point counts, timing)
    """
    print("=" * 80)
    print("Point Cloud Visualization - 3 Windows")
    print("=" * 80)
    print("Windows:")
    if enable_femto_vis:
        print("  1. Femto Bolt (Camera Frame) - ENABLED")
    else:
        print("  1. Femto Bolt (Camera Frame) - DISABLED")
    if enable_d405_vis:
        print("  2. D405 (Camera Frame) - ENABLED")
    else:
        print("  2. D405 (Camera Frame) - DISABLED")
    if enable_blending_vis:
        print("  3. Blending (Base Frame) - ENABLED")
    else:
        print("  3. Blending (Base Frame) - DISABLED")
    print("")
    print("Press 'Q' in any window to quit")
    print("=" * 80)

    with SharedMemoryManager() as shm_manager:
        # ========== 1. 카메라 초기화 ==========
        print("\n[1/4] Initializing Femto Bolt...")
        femto = SingleOrbbec(
            shm_manager=shm_manager,
            rgb_resolution=(1280, 720),
            put_fps=30,
            mode="C2D"  # Color to Depth aligned
        )

        print("[2/4] Initializing D405...")
        d405 = SingleRealSense(
            shm_manager=shm_manager,
            resolution=(424, 240),
            put_fps=60,
            # SDK filters disabled - OpenCV filter will be applied in BlendingProcessor
            enable_threshold_filter=True,
            enable_spatial_filter=True
        )

        # ========== 2. 로봇 인터페이스 (read-only) ==========
        print("[3/4] Connecting to PiPER (read-only)...")
        piper = None
        if PIPER_AVAILABLE:
            try:
                piper = C_PiperInterface_V2("can_slave")
                piper.ConnectPort()
                print("✓ Connected to PiPER (read-only mode)")
            except Exception as e:
                print(f"Warning: Could not connect to PiPER: {e}")
                print("Using dummy pose.")
                piper = None

        # ========== 3. Blending Processor 초기화 ==========
        print("[4/4] Initializing Blending Processor (Variance Filter OFF)...")
        blending_processor = BlendingProcessor(
            femto_camera=femto,
            d405_camera=d405,
            piper_interface=piper,
        )

        # ========== 4. 카메라 시작 ==========
        print("\nStarting cameras...")
        femto.start(wait=False)
        d405.start(wait=False)

        print("Waiting for cameras to be ready...")
        femto.start_wait()
        d405.start_wait()
        print("✓ All devices ready!")

        # ========== 5. Visualizers 생성 ==========
        print("\nCreating visualizers...")

        vis_femto = None
        vis_d405 = None
        vis_blending = None

        # Femto visualizer (left)
        if enable_femto_vis:
            vis_femto = PointCloudVisualizer(
                window_name="Femto Bolt - Camera Frame (mm)",
                width=640, height=480,
                left=0, top=100,
                show_coordinate=True,
                coordinate_size=200.0  # mm
            )
            vis_femto.set_view(
                lookat=[200, 0, 200],
                front=[0, -1, -0.5],
                up=[0, 0, 1],
                zoom=0.2
            )

        # D405 visualizer (middle)
        if enable_d405_vis:
            vis_d405 = PointCloudVisualizer(
                window_name="D405 - Camera Frame (m)",
                width=640, height=480,
                left=650, top=100,
                show_coordinate=True,
                coordinate_size=0.15  # m
            )
            vis_d405.set_view(
                lookat=[0.2, 0.0, 0.2],
                front=[0, -1, -0.5],
                up=[0, 0, 1],
                zoom=0.5
            )

        # Blending visualizer (right) - Base frame
        if enable_blending_vis:
            vis_blending = PointCloudVisualizer(
                window_name="Blending - Base Frame (m)",
                width=640, height=480,
                left=1300, top=100,
                show_coordinate=True,
                coordinate_size=0.15  # m
            )
            vis_blending.set_view(
                lookat=[0.2, 0.0, 0.2],
                front=[0, -1, -0.5],
                up=[0, 0, 1],
                zoom=0.2
            )

        # Default workspace bounds
        workspace_bounds = [
            [0.000, 0.715],     # X range (m)
            [-0.400, 0.350],    # Y range (m)
            [-0.100, 0.400]     # Z range (m)
        ]

        # ========== 6. Base frame 변환 함수 (간단한 numpy 변환) ==========
        def transform_to_base(pc_mm_rgb01):
            """
            Femto camera frame (mm, RGB 0-1) → Base frame (m, RGB 0-1)

            Args:
                pc_mm_rgb01: (N, 6) numpy array, XYZ(mm) + RGB(0-1)

            Returns:
                (N, 6) numpy array, XYZ(m) + RGB(0-1)
            """
            if len(pc_mm_rgb01) == 0:
                return np.zeros((0, 6), dtype=np.float32)

            # 1. XYZ 변환: mm → m
            xyz_mm = pc_mm_rgb01[:, :3]
            xyz_m = xyz_mm / 1000.0

            # 2. Homogeneous coordinates (N, 4)
            ones = np.ones((len(xyz_m), 1), dtype=np.float32)
            xyz_homo = np.hstack([xyz_m, ones])

            # 3. Transform: Femto cam → Base
            T = blending_processor.femto_cam_to_base  # (4, 4)
            xyz_base_homo = (T @ xyz_homo.T).T  # (N, 4)
            xyz_base = xyz_base_homo[:, :3]

            # 4. RGB는 이미 0-1 범위이므로 그대로 사용
            rgb = pc_mm_rgb01[:, 3:6]

            return np.hstack([xyz_base, rgb]).astype(np.float32)

        def crop_workspace(points):
            if len(points) == 0:
                return points

            mask = (
                (points[:, 0] >= workspace_bounds[0][0]) &
                (points[:, 0] <= workspace_bounds[0][1]) &
                (points[:, 1] >= workspace_bounds[1][0]) &
                (points[:, 1] <= workspace_bounds[1][1]) &
                (points[:, 2] >= workspace_bounds[2][0]) &
                (points[:, 2] <= workspace_bounds[2][1])
            )

            cropped_points = points[mask]

            return cropped_points

        def apply_ror_filter(points, nb_points=12, radius=0.01):
            """
            ROR (Radius Outlier Removal) 필터 적용
            Args:
                points: (N, 6) array [x, y, z, r, g, b]
                nb_points: 반경 내 최소 점 개수
                radius: 검색 반경 (meters)
            Returns:
                filtered_points: (M, 6) array
            """
            if len(points) == 0:
                return points

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            _, inlier_indices = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
            return points[inlier_indices]

        # ========== 7. 메인 루프 ==========
        print("\nVisualization started. Press 'Q' to quit.\n")

        # Import pipeline debug utilities
        from pcdp.common.pipeline_debug import PipelineDebugInfo, print_pipeline_debug

        # Flags for enabling/disabling pipeline stages
        enable_d405 = True
        enable_voxel = True
        enable_blending = True
        enable_gradient = blending_processor.enable_composite_filter
        enable_crop = True
        enable_ror = False  # Set to False to disable ROR filter

        frame_count = 0
        try:
            while True:
                t_start = time.perf_counter()

                # Process: 데이터 가져오기 + Blending
                result = blending_processor.process(verbose=verbose)

                femto_pc_raw = result['femto_pc']      # (N, 6) XYZ(mm) + RGB(0-1)
                d405_pc_raw = result['d405_pc']        # (M, 6) XYZ(m) + RGB(0-1)
                blended_pc_raw = result['blended_pc']  # (K, 6) XYZ(mm) + RGB(0-1)
                debug_info = result.get('debug', None)  # verbose일 때만 존재

                # 6. Base transform: Blending 점군을 Base frame (m)으로 변환
                t_transform_start = time.perf_counter()
                blended_pc = transform_to_base(blended_pc_raw)  # (K, 6) XYZ(m) + RGB(0-1)
                n_after_transform = len(blended_pc)
                t_transform_end = time.perf_counter()

                # 7. Workspace crop
                t_crop_start = time.perf_counter()
                if enable_crop:
                    blended_pc = crop_workspace(blended_pc)
                n_after_crop = len(blended_pc)
                t_crop_end = time.perf_counter()

                # 8. ROR 필터 적용 (Workspace crop 후)
                t_ror_start = time.perf_counter()
                if enable_ror:
                    blended_pc = apply_ror_filter(blended_pc, nb_points=12, radius=0.01)
                n_after_ror = len(blended_pc)
                t_ror_end = time.perf_counter()

                t_process = time.perf_counter()

                # Femto: Camera frame (mm) 그대로 시각화
                femto_valid = femto_pc_raw[femto_pc_raw[:, 2] > 0.0]
                if vis_femto is not None and len(femto_valid) > 0:
                    vis_femto.update(
                        femto_valid[:, :3],
                        np.clip(femto_valid[:, 3:6], 0, 1)  # 이미 0-1 범위
                    )

                # D405: Camera frame (m) 그대로 시각화
                d405_valid = d405_pc_raw[d405_pc_raw[:, 2] > 0.0]
                if vis_d405 is not None and len(d405_valid) > 0:
                    vis_d405.update(
                        d405_valid[:, :3],
                        np.clip(d405_valid[:, 3:6], 0, 1)
                    )

                # Blending: 이미 Base frame (m)으로 변환됨, 그대로 시각화
                if vis_blending is not None and len(blended_pc) > 0:
                    vis_blending.update(
                        blended_pc[:, :3],
                        np.clip(blended_pc[:, 3:6], 0, 1)  # 이미 0-1 범위
                    )
                
                t_viz = time.perf_counter()

                # 통계 출력 (30프레임마다, verbose일 때만)
                if verbose and frame_count % 30 == 0 and debug_info is not None:
                    t_process_ms = (t_process - t_start) * 1000
                    t_viz_ms = (t_viz - t_process) * 1000
                    t_total_ms = (t_viz - t_start) * 1000

                    # Create debug info object
                    pipeline_info = PipelineDebugInfo(
                        # Point counts
                        n_femto=debug_info['n_femto_raw'],
                        n_d405_raw=debug_info['n_d405_raw'],
                        n_d405_voxel=debug_info['n_d405_voxel'],
                        n_blended=debug_info['n_blended'],
                        n_gradient=debug_info.get('n_after_filter', debug_info['n_blended']),
                        n_transform=n_after_transform,
                        n_crop=n_after_crop,
                        n_ror=n_after_ror,
                        # Timing (ms)
                        t_femto=debug_info['t_femto_ms'],
                        t_d405=debug_info['t_d405_ms'],
                        t_voxel=debug_info['t_voxel_ms'],
                        t_blend=debug_info['t_blend_ms'],
                        t_gradient=debug_info.get('t_composite_ms', 0.0),
                        t_transform=(t_transform_end - t_transform_start) * 1000,
                        t_crop=(t_crop_end - t_crop_start) * 1000,
                        t_ror=(t_ror_end - t_ror_start) * 1000,
                        # Process/Viz/Total timing
                        t_process=t_process_ms,
                        t_viz=t_viz_ms,
                        t_total=t_total_ms,
                        # Flags
                        enable_d405=enable_d405,
                        enable_voxel=enable_voxel,
                        enable_blending=enable_blending,
                        enable_gradient=enable_gradient,
                        enable_crop=enable_crop,
                        enable_ror=enable_ror,
                        # Frame count
                        frame_count=frame_count
                    )
                    print_pipeline_debug(pipeline_info)

                # 이벤트 폴링 (창 닫힘 확인)
                if vis_femto is not None:
                    if not vis_femto.poll_events():
                        break
                if vis_d405 is not None:
                    if not vis_d405.poll_events():
                        break
                if vis_blending is not None:
                    if not vis_blending.poll_events():
                        break

                frame_count += 1

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            # ========== 8. 정리 ==========
            print("\nCleaning up...")

            if vis_femto is not None:
                vis_femto.destroy()
            if vis_d405 is not None:
                vis_d405.destroy()
            if vis_blending is not None:
                vis_blending.destroy()

            print("Shutting down cameras...")
            femto.stop()
            d405.stop()

            femto.join()
            d405.join()

            # PiPER 연결 해제
            if piper is not None:
                try:
                    piper.DisconnectPort()
                    print("Disconnected from PiPER")
                except:
                    pass

            print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Visualization")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Enable verbose debug output (point counts, timing)")
    parser.add_argument('--no-femto', action='store_true',
                        help="Disable Femto Bolt visualization")
    parser.add_argument('--no-d405', action='store_true',
                        help="Disable D405 visualization")
    parser.add_argument('--no-blending', action='store_true',
                        help="Disable Blending visualization")
    args = parser.parse_args()

    main(
        enable_femto_vis=not args.no_femto,
        enable_d405_vis=not args.no_d405,
        enable_blending_vis=not args.no_blending,
        verbose=args.verbose
    )
