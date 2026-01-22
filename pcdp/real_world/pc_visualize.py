#!/usr/bin/env python3
"""
Point Cloud Visualization - 3 Windows
Femto Bolt, D405, Blending 동시 시각화
"""
import sys
import numpy as np
import open3d as o3d
from multiprocessing.managers import SharedMemoryManager

# PCDP imports
from pcdp.real_world.single_orbbec import SingleOrbbec
from pcdp.real_world.single_realsense import SingleRealSense
from pcdp.real_world.blending_processor import BlendingProcessor
from pcdp.real_world.pc_visualizer import PointCloudVisualizer
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from pcdp.real_world.teleoperation_piper import TeleoperationPiper
from pcdp.real_world.piper_interpolation_controller import PiperInterpolationController
import pcdp.common.mono_time as mono_time

# PiPER SDK (optional)
try:
    from piper_sdk import C_PiperInterface_V2
    PIPER_AVAILABLE = True
except ImportError:
    print("Warning: PiPER SDK not available. Using dummy pose.")
    PIPER_AVAILABLE = False


def filter_pointcloud_by_depth(pointcloud, depth_image, filtered_depth, intrinsics):
    """
    Point cloud를 depth image에 투영하고, filtered_depth에서 살아남은 점만 반환.

    Args:
        pointcloud: (N, 6) XYZ(mm) + RGB
        depth_image: (H, W) original depth image (mm)
        filtered_depth: (H, W) variance filtered depth image (mm)
        intrinsics: (fx, fy, cx, cy)

    Returns:
        filtered_pc: (M, 6) filtered point cloud
    """
    if len(pointcloud) == 0:
        return pointcloud

    fx, fy, cx, cy = intrinsics
    h, w = depth_image.shape

    # Point cloud의 z > 0인 점만 처리
    valid_mask = pointcloud[:, 2] > 0
    xyz = pointcloud[valid_mask, :3]

    # 3D → 2D 투영
    u = (fx * xyz[:, 0] / xyz[:, 2] + cx).astype(np.int32)
    v = (fy * xyz[:, 1] / xyz[:, 2] + cy).astype(np.int32)

    # 이미지 경계 내 점만
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # filtered_depth에서 살아남은 점만 유지
    keep_mask = np.zeros(len(xyz), dtype=bool)
    keep_mask[in_bounds] = filtered_depth[v[in_bounds], u[in_bounds]] > 0

    # 원본 pointcloud에서 유효한 점 중 keep된 점만 반환
    valid_indices = np.where(valid_mask)[0]
    final_indices = valid_indices[keep_mask]

    return pointcloud[final_indices]


def main(
    # Visualization flags
    enable_femto_vis=True,
    enable_d405_vis=True,
    enable_blending_vis=True,
    # Pipeline flags
    enable_variance_filter=True,
    enable_femto_variance_filter=True,
    enable_crop=True,
    enable_ror=False,
    # PCM flags (각 카메라별)
    enable_pcm_femto=True,
    enable_pcm_d405=False,  # 추후 구현
    enable_teleop=True,
):
    """
    Args:
        enable_femto_vis: Femto Bolt 시각화 활성화
        enable_d405_vis: D405 시각화 활성화
        enable_blending_vis: Blending 시각화 활성화
        enable_variance_filter: Variance 기반 outlier 필터 활성화
        enable_crop: Workspace crop 활성화
        enable_ror: ROR (Radius Outlier Removal) 필터 활성화
        enable_pcm_femto: Femto PCM temporal memory 활성화
        enable_pcm_d405: D405 PCM temporal memory 활성화 (추후 구현)
        enable_teleop: Teleoperation 활성화 (master→slave)
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
    print("Pipeline:")
    print(f"  - Variance Filter: {'ON' if enable_variance_filter else 'OFF'}")
    print(f"  - Crop: {'ON' if enable_crop else 'OFF'}")
    print(f"  - ROR: {'ON' if enable_ror else 'OFF'}")
    print(f"  - PCM Femto: {'ON' if enable_pcm_femto else 'OFF'}")
    print(f"  - PCM D405: {'ON' if enable_pcm_d405 else 'OFF'}")
    print(f"  - Teleop: {'ON' if enable_teleop else 'OFF'}")
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

        # ========== 2. 로봇 인터페이스 ==========
        piper = None
        teleop = None
        robot_controller = None

        if enable_teleop:
            print("[3/4] Initializing Teleoperation (master→slave)...")
            try:
                # Master arm reader (can_master)
                teleop = TeleoperationPiper(shm_manager=shm_manager)
                teleop.start(wait=True)
                print("✓ TeleoperationPiper (master) ready")

                # Slave arm controller (can_slave)
                robot_controller = PiperInterpolationController(
                    shm_manager=shm_manager,
                    frequency=200,
                    mode="EndPose"
                )
                robot_controller.start(wait=True)
                print("✓ PiperInterpolationController (slave) ready")

                # Slave arm pose 읽기용 read-only 연결 (BlendingProcessor용)
                if PIPER_AVAILABLE:
                    try:
                        piper = C_PiperInterface_V2("can_slave")
                        piper.ConnectPort()
                        print("✓ Connected to PiPER (read-only for BlendingProcessor)")
                    except Exception as e:
                        print(f"Warning: Could not connect to PiPER for pose reading: {e}")
                        piper = None
            except Exception as e:
                print(f"Warning: Teleoperation init failed: {e}")
                print("Falling back to read-only mode.")
                enable_teleop = False
                if teleop is not None:
                    try:
                        teleop.stop()
                    except:
                        pass
                    teleop = None
                if robot_controller is not None:
                    try:
                        robot_controller.stop()
                    except:
                        pass
                    robot_controller = None

        if not enable_teleop:
            print("[3/4] Connecting to PiPER (read-only)...")
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
        variance_status = "ON" if enable_variance_filter else "OFF"
        print(f"[4/4] Initializing Blending Processor (Variance Filter {variance_status})...")
        blending_processor = BlendingProcessor(
            femto_camera=femto,
            d405_camera=d405,
            piper_interface=piper,
            enable_variance_filter=enable_variance_filter,
        )

        # ========== 4. 카메라 시작 ==========
        print("\nStarting cameras...")
        femto.start(wait=False)
        d405.start(wait=False)

        print("Waiting for cameras to be ready...")
        femto.start_wait()
        d405.start_wait()
        print("✓ All devices ready!")

        # ========== 4.5 PCM Preprocessor 초기화 (Femto용) ==========
        # PCM_stack.yaml config와 동일한 설정
        pcm_femto = None
        if enable_pcm_femto:
            print("\nInitializing PCM for Femto...")
            pcm_femto = PointCloudPreprocessor(
                # 이미 수동으로 transform/crop 했으므로 비활성화
                enable_transform=False,
                enable_cropping=False,
                enable_sampling=False,
                enable_filter=False,
                # PCM temporal 설정 (PCM_stack.yaml과 동일)
                enable_temporal=True,
                export_mode='fused',
                temporal_voxel_size=0.005,
                temporal_decay=0.95,
                use_cuda=True,
                # Occlusion prune 설정
                enable_occlusion_prune=True,
                depth_width=320,
                depth_height=288,
                K_depth=[
                    [252.69204711914062, 0.0, 166.12030029296875],
                    [0.0, 252.65277099609375, 176.21173095703125],
                    [0.0, 0.0, 1.0]
                ],
                extrinsics_matrix=[
                    [0.007131, -0.91491, 0.403594, 0.05116],
                    [-0.994138, 0.003833, 0.02656, -0.00918],
                    [-0.025717, -0.403641, -0.914552, 0.50821],
                    [0., 0., 0., 1.]
                ],
                miss_prune_frames=20,
                miss_min_age=2,
            )
            print("✓ PCM Femto initialized")

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

        # ========== 6. 유틸리티 함수 (일반화: N,6 또는 N,7 모두 지원) ==========
        def transform_to_base(pc_mm):
            """
            Femto camera frame (mm) → Base frame (m)
            Args:
                pc_mm: (N, K) array, XYZ(mm) + 나머지 열 (K=6 또는 7)
            Returns:
                (N, K) array, XYZ(m) + 나머지 열 유지
            """
            if len(pc_mm) == 0:
                return pc_mm

            xyz_m = pc_mm[:, :3] / 1000.0
            ones = np.ones((len(xyz_m), 1), dtype=np.float32)
            xyz_homo = np.hstack([xyz_m, ones])

            T = blending_processor.femto_cam_to_base
            xyz_base = (T @ xyz_homo.T).T[:, :3]

            return np.hstack([xyz_base, pc_mm[:, 3:]]).astype(np.float32)

        def crop_workspace(points):
            """Workspace crop (N, K) → XYZ 기준 crop"""
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
            return points[mask]

        def apply_ror_filter(points, nb_points=12, radius=0.01):
            """ROR (Radius Outlier Removal) 필터 (N, K) → XYZ 기준 필터"""
            if len(points) == 0:
                return points

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            _, inlier_indices = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
            return points[inlier_indices]

        # ========== 7. 메인 루프 ==========
        print("\nVisualization started. Press 'Q' to quit.\n")
        if enable_teleop:
            print("Teleoperation active: Move master arm to control slave arm.\n")

        dt = 1.0 / 30.0  # 30Hz control rate

        try:
            while True:
                # Teleoperation: 마스터 암 → 슬레이브 암
                if enable_teleop and teleop is not None and robot_controller is not None:
                    target_pose = teleop.get_motion_state()  # [x,y,z,rx,ry,rz,gripper]
                    robot_controller.schedule_waypoint(
                        pose=target_pose,
                        target_time=mono_time.now_s() + dt
                    )

                # ===== 1. Blended + Labeled + Variance Filtered 데이터 가져오기 =====
                result = blending_processor.process(return_labeled=True)
                blended_labeled = result['blended_pc']  # (N, 7) XYZ(mm) + RGB(0-1) + label

                femto_pc_raw = result['femto_pc']       # For visualization (camera frame)
                d405_pc_raw = result['d405_pc']         # For visualization (camera frame)

                # ===== 2. Base transform (mm → m) =====
                blended_base = transform_to_base(blended_labeled)  # (N, 7)

                # ===== 3. Workspace crop =====
                if enable_crop:
                    blended_base = crop_workspace(blended_base)

                # ===== 4. ROR filter =====
                if enable_ror:
                    blended_base = apply_ror_filter(blended_base)

                # ===== 5. Label로 분리 =====
                if len(blended_base) > 0:
                    femto_mask = blended_base[:, 6] == 0  # label=0: Femto
                    d405_mask = blended_base[:, 6] == 1   # label=1: D405

                    femto_filtered = blended_base[femto_mask, :6]  # (N, 6) xyz + rgb
                    d405_filtered = blended_base[d405_mask, :6]    # (M, 6) xyz + rgb
                else:
                    femto_filtered = np.zeros((0, 6), dtype=np.float32)
                    d405_filtered = np.zeros((0, 6), dtype=np.float32)

                # ===== 6. PCM (Femto만 적용) =====
                if enable_pcm_femto and pcm_femto is not None and len(femto_filtered) > 0:
                    pcm_output = pcm_femto.process(femto_filtered)
                    if len(pcm_output) > 0:
                        femto_pcm = pcm_output[:, :6]  # P_f + P_f' (현재 + 잔존)
                    else:
                        femto_pcm = femto_filtered
                else:
                    femto_pcm = femto_filtered

                # ===== 7. 최종 합치기: P_f + P_f' + P_d =====
                parts = []
                if len(femto_pcm) > 0:
                    parts.append(femto_pcm)
                if len(d405_filtered) > 0:
                    parts.append(d405_filtered)
                blended_pc = np.vstack(parts) if parts else np.zeros((0, 6), dtype=np.float32)

                # Femto: Camera frame (mm) 그대로 시각화
                if enable_femto_variance_filter:                                                                   
                    # 1. Femto depth image와 intrinsics 가져오기                                                   
                    femto_data = femto.get(k=1)                                                                    
                    femto_depth = femto_data['depth_image'][-1]                                                    
                    femto_intrinsics = femto_data['intrinsics'][-1]                                                
                                                                                                                    
                    # 2. Depth image에 variance filter 적용                                                        
                    filtered_depth = blending_processor.apply_variance_filter(femto_depth)                         
                                                                                                                    
                    # 3. filtered_depth 기준으로 point cloud 필터링                                                
                    femto_valid = filter_pointcloud_by_depth(                                                      
                        femto_pc_raw, femto_depth, filtered_depth, femto_intrinsics                                
                    )                                                                                              
                else:                                                                                              
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

            # Teleoperation 종료
            if robot_controller is not None:
                print("Stopping robot controller...")
                try:
                    robot_controller.stop()
                except:
                    pass

            if teleop is not None:
                print("Stopping teleoperation...")
                try:
                    teleop.stop()
                except:
                    pass

            # PiPER 연결 해제 (read-only mode)
            if piper is not None:
                try:
                    piper.DisconnectPort()
                    print("Disconnected from PiPER")
                except:
                    pass

            print("Done!")


if __name__ == "__main__":
    main()