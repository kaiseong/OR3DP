import pathlib
import numpy as np
import open3d as o3d
import cv2
import time

from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from pcdp.real_world.single_realsense import SingleRealSense
from pcdp.common.replay_buffer import ReplayBuffer
from pcdp.common.RISE_transformation import rot_trans_mat

path_dataset = '/home/moai/OR3DP/data/cube_stack/recorder_data'

robot_to_base = np.array([
    [1.000, 0.000, 0.000, -0.040],
    [0.000, 1.000, 0.000, -0.290],
    [0.000, 0.000, 1.000, -0.030],
    [0.000, 0.000, 0.000,  1.000]
])

ee_to_d405 = np.array([
    [0.011562517413824459, -0.9045824108766296, 0.4261419600598841, -0.0630571160287582],
    [0.9999276419016558, 0.009045119479890396, -0.007930748677719701, -0.006957758124793954],
    [0.003319510814811938, 0.42620282465772, 0.9046215413650787, 0.055453185424837834],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

femto_cam_to_base = np.array([
    [  0.007131,  -0.91491,    0.403594,   0.05116],
    [ -0.994138,   0.003833,   0.026560,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552,   0.50821],
    [  0.,         0.,         0.,         1.     ]
], dtype=np.float64)


def visualize_variance_filter(depth_image, kernel_size=5, variance_threshold=50.0):
    """
    Visualize variance filter effect on depth image.

    - Grayscale: depth (closer = brighter)
    - Red: pixels that would be filtered out (high variance)

    Returns:
        vis_image: (H, W, 3) BGR image for display
        filtered_count: number of filtered pixels
        total_valid: total valid pixels before filtering
    """
    depth = depth_image.astype(np.float32)
    valid_mask = depth > 0
    total_valid = np.sum(valid_mask)

    depth_zero_filled = np.where(valid_mask, depth, 0)

    # Calculate variance
    ksize = (kernel_size, kernel_size)
    depth_sum = cv2.boxFilter(depth_zero_filled, -1, ksize, normalize=False)
    valid_count = cv2.boxFilter(valid_mask.astype(np.float32), -1, ksize, normalize=False)
    valid_count_safe = np.maximum(valid_count, 1)

    mean = depth_sum / valid_count_safe
    depth_sq = depth_zero_filled ** 2
    depth_sq_sum = cv2.boxFilter(depth_sq, -1, ksize, normalize=False)
    sq_mean = depth_sq_sum / valid_count_safe
    variance = sq_mean - mean ** 2

    # Pixels to keep vs filter
    keep_mask = (variance < variance_threshold) & valid_mask
    filter_mask = (variance >= variance_threshold) & valid_mask
    filtered_count = np.sum(filter_mask)

    # Create grayscale depth visualization
    if total_valid > 0:
        valid_depth = depth[valid_mask]
        min_d, max_d = valid_depth.min(), valid_depth.max()
        if max_d > min_d:
            normalized = (depth - min_d) / (max_d - min_d)
        else:
            normalized = np.zeros_like(depth)
        gray = (normalized * 255).astype(np.uint8)
    else:
        gray = np.zeros(depth.shape, dtype=np.uint8)

    # Convert to BGR
    vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Mark filtered pixels in red (BGR: 0, 0, 255)
    vis_image[filter_mask] = [0, 0, 255]

    # Mark invalid pixels in black
    vis_image[~valid_mask] = [0, 0, 0]

    return vis_image, filtered_count, total_valid


def depth_to_pointcloud(depth_image, color_image, intrinsics, depth_scale=0.0001):
    """
    Convert depth image to point cloud with color.

    Args:
        depth_image: (H, W) uint16 depth image in mm
        color_image: (H, W, 3) RGB color image
        intrinsics: [fx, fy, cx, cy]
        depth_scale: depth to meters scale (default 0.001 for mm to m)

    Returns:
        points: (N, 6) array [x, y, z, r, g, b], xyz in meters, rgb in [0,1]
    """
    H, W = depth_image.shape
    fx, fy, cx, cy = intrinsics

    # Create pixel coordinate grid
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # Get valid depth mask
    valid_mask = depth_image > 0

    # Convert depth to meters
    z = depth_image.astype(np.float32) * depth_scale

    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack coordinates
    xyz = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    # Get colors normalized to [0, 1]
    rgb = color_image.astype(np.float32) / 255.0  # (H, W, 3)

    # Combine xyz and rgb
    points_full = np.concatenate([xyz, rgb], axis=-1)  # (H, W, 6)

    # Filter valid points
    points = points_full[valid_mask]  # (N, 6)

    return points


def get_d405_dynamic_extrinsics(robot_eef_pose):
    """
    Compute D405 camera to base frame transformation (dynamic extrinsics).

    Args:
        robot_eef_pose: (6,) [x, y, z, rx, ry, rz] end-effector pose in meters and radians

    Returns:
        d405_to_base: (4, 4) D405 camera to base frame transform
    """
    # EE pose to transformation matrix
    manipulator_to_ee = rot_trans_mat(robot_eef_pose[:3], robot_eef_pose[3:6])

    # D405 -> EE -> Robot -> Base
    d405_to_base = robot_to_base @ manipulator_to_ee @ ee_to_d405

    return d405_to_base


class DatasetVisualizer:
    def __init__(
        self,
        dataset_path,
        episode_idx=100,
        # Femto preprocessing
        enable_femto_transform=True,
        enable_femto_cropping=True,
        enable_femto_variance_filter=True,
        femto_variance_kernel_size=5,
        femto_variance_threshold=50.0,
        workspace_bounds=None,
        # D405 settings
        enable_wrist_camera=True,
        enable_d405_transform=True,
        enable_d405_variance_filter = True,
        d405_variance_kernel_size=5,
        d405_variance_threshold = 50.0,
        # Visualization flags
        show_femto=True,
        show_d405=True,
        show_fusion=True,
        # Playback settings
        frame_mode=False,
        playback_fps=10,
    ):
        self.dataset_path = pathlib.Path(dataset_path)
        self.episode_idx = episode_idx

        # Femto settings
        self.enable_femto_transform = enable_femto_transform
        self.enable_femto_cropping = enable_femto_cropping
        self.enable_femto_variance_filter = enable_femto_variance_filter
        self.femto_variance_kernel_size = femto_variance_kernel_size
        self.femto_variance_threshold = femto_variance_threshold

        # Workspace bounds
        if workspace_bounds is None:
            self.workspace_bounds = [
                [0.132, 0.715],    # X
                [-0.400, 0.350],   # Y
                [-0.100, 0.400]    # Z
            ]
        else:
            self.workspace_bounds = workspace_bounds

        # D405 settings
        self.enable_wrist_camera = enable_wrist_camera
        self.enable_d405_transform = enable_d405_transform
        self.enable_d405_variance_filter = enable_d405_variance_filter
        self.d405_variance_kernel_size = d405_variance_kernel_size
        self.d405_variance_threshold = d405_variance_threshold

        # Visualization flags
        self.show_femto = show_femto
        self.show_d405 = show_d405 and enable_wrist_camera
        self.show_fusion = show_fusion and enable_wrist_camera

        # Playback settings
        self.frame_mode = frame_mode
        self.playback_fps = playback_fps
        self.frame_interval = 1.0 / playback_fps

        # State
        self.current_frame = 0
        self.paused = False
        self.running = True

        # Load data
        self._load_episode_data()

        # Initialize preprocessor for Femto
        self.femto_preprocessor = PointCloudPreprocessor(
            enable_sampling=False,
            enable_transform=enable_femto_transform,
            extrinsics_matrix=femto_cam_to_base,
            enable_cropping=enable_femto_cropping,
            workspace_bounds=self.workspace_bounds,
            enable_wrist_camera=False,
            enable_filter=enable_femto_variance_filter,
            variance_kernel_size=femto_variance_kernel_size,
            variance_threshold=femto_variance_threshold,
            enable_temporal=False,
        )

        # Initialize preprocessor for D405 wrist camera
        if self.enable_wrist_camera:
            self.wrist_preprocessor = PointCloudPreprocessor(
                enable_sampling=False,
                enable_transform=enable_d405_transform,
                extrinsics_matrix=None,  # Dynamic extrinsics - passed per frame
                enable_cropping=False,   # No workspace cropping for wrist
                enable_wrist_camera=False,
                enable_filter=False,     # Variance filter applied on depth image
                enable_temporal=False,
            )

    def _load_episode_data(self):
        """Load episode data from zarr."""
        episode_dir = self.dataset_path / f'episode_{self.episode_idx:04d}'
        obs_zarr_path = episode_dir / 'obs_replay_buffer.zarr'

        if not obs_zarr_path.exists():
            raise FileNotFoundError(f"Episode not found: {obs_zarr_path}")

        obs_buffer = ReplayBuffer.create_from_path(str(obs_zarr_path), mode='r')

        # Load Femto data
        self.femto_pointclouds = obs_buffer['pointcloud'][:]

        # Load D405 data if enabled
        if self.enable_wrist_camera:
            self.d405_depth_images = obs_buffer['eef_depth_image'][:]
            self.d405_color_images = obs_buffer['eef_color_image'][:]
            self.d405_intrinsics = obs_buffer['eef_intrinsics'][:]
            self.robot_eef_poses = obs_buffer['robot_eef_pose'][:]

        self.num_frames = len(self.femto_pointclouds)
        print(f"Loaded episode {self.episode_idx} with {self.num_frames} frames")

    def _process_femto_frame(self, frame_idx):
        """Process Femto Bolt point cloud for a frame."""
        raw_pc = self.femto_pointclouds[frame_idx]
        processed_pc = self.femto_preprocessor.process(raw_pc)
        return processed_pc

    def _process_d405_frame(self, frame_idx):
        """Process D405 depth image to point cloud for a frame."""
        if not self.enable_wrist_camera:
            return None

        depth_image = self.d405_depth_images[frame_idx]
        color_image = self.d405_color_images[frame_idx]
        intrinsics = self.d405_intrinsics[frame_idx]
        robot_pose = self.robot_eef_poses[frame_idx]

        # Apply variance filter to depth image (before point cloud conversion)
        if self.enable_d405_variance_filter:
            depth_image = SingleRealSense.apply_variance_filter(
                depth_image,
                self.d405_variance_kernel_size,
                self.d405_variance_threshold
            )

        # Convert depth to point cloud (in camera frame)
        d405_pc = depth_to_pointcloud(depth_image, color_image, intrinsics)

        if len(d405_pc) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # Transform to base frame using process_wrist with dynamic extrinsics
        if self.enable_d405_transform:
            dynamic_extrinsics = get_d405_dynamic_extrinsics(robot_pose)
            d405_pc = self.wrist_preprocessor.process_wrist(d405_pc, dynamic_extrinsics)

        return d405_pc

    def _create_visualization_geometry(self, frame_idx):
        """Create Open3D geometry for visualization."""
        geometries = []

        # Process Femto
        femto_pc = self._process_femto_frame(frame_idx)

        # Process D405
        d405_pc = self._process_d405_frame(frame_idx) if self.enable_wrist_camera else None

        # Create point clouds for visualization
        if self.show_femto and len(femto_pc) > 0:
            pcd_femto = o3d.geometry.PointCloud()
            pcd_femto.points = o3d.utility.Vector3dVector(femto_pc[:, :3])
            if femto_pc.shape[1] >= 6:
                pcd_femto.colors = o3d.utility.Vector3dVector(femto_pc[:, 3:6])
            geometries.append(('femto', pcd_femto))

        if self.show_d405 and d405_pc is not None and len(d405_pc) > 0:
            pcd_d405 = o3d.geometry.PointCloud()
            pcd_d405.points = o3d.utility.Vector3dVector(d405_pc[:, :3])
            if d405_pc.shape[1] >= 6:
                pcd_d405.colors = o3d.utility.Vector3dVector(d405_pc[:, 3:6])
            geometries.append(('d405', pcd_d405))

        if self.show_fusion and len(femto_pc) > 0:
            # Simple fusion: concatenate both point clouds
            if d405_pc is not None and len(d405_pc) > 0:
                fusion_pc = np.vstack([femto_pc, d405_pc])
            else:
                fusion_pc = femto_pc

            pcd_fusion = o3d.geometry.PointCloud()
            pcd_fusion.points = o3d.utility.Vector3dVector(fusion_pc[:, :3])
            if fusion_pc.shape[1] >= 6:
                pcd_fusion.colors = o3d.utility.Vector3dVector(fusion_pc[:, 3:6])
            geometries.append(('fusion', pcd_fusion))

        return geometries

    def _key_callback(self, vis, action, mods):
        """Handle keyboard input."""
        if action != 1:  # Key press only
            return False
        return False

    def run(self):
        """Run the visualization."""
        # Count active views
        active_views = sum([self.show_femto, self.show_d405, self.show_fusion])
        if active_views == 0:
            print("No views enabled. Enable at least one of: show_femto, show_d405, show_fusion")
            return

        # Create visualizers for each view
        visualizers = {}
        window_width = 800
        window_height = 600

        view_names = []
        if self.show_femto:
            view_names.append('femto')
        if self.show_d405:
            view_names.append('d405')
        if self.show_fusion:
            view_names.append('fusion')

        for i, name in enumerate(view_names):
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(
                window_name=f'{name.upper()} Point Cloud',
                width=window_width,
                height=window_height,
                left=50 + i * (window_width + 10),
                top=50
            )

            # Register key callbacks
            vis.register_key_callback(ord('Q'), lambda v: self._quit_callback())
            vis.register_key_callback(ord(' '), lambda v: self._toggle_pause())
            vis.register_key_callback(ord('N'), lambda v: self._next_frame())
            vis.register_key_callback(ord('P'), lambda v: self._prev_frame())
            vis.register_key_callback(ord('R'), lambda v: self._reset_frame())

            visualizers[name] = vis

        print("\n=== Dataset Visualizer ===")
        print(f"Episode: {self.episode_idx}, Frames: {self.num_frames}")
        print(f"Mode: {'Frame-by-frame' if self.frame_mode else 'Auto-play'}")
        print("\nControls:")
        print("  Space: Pause/Resume")
        print("  N: Next frame")
        print("  P: Previous frame")
        print("  R: Reset to first frame")
        print("  Q: Quit")
        print("  Mouse: Rotate/Zoom view")
        print("="*30 + "\n")

        # Initial geometry
        point_clouds = {}
        for name in view_names:
            point_clouds[name] = o3d.geometry.PointCloud()
            visualizers[name].add_geometry(point_clouds[name])

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        for name in view_names:
            visualizers[name].add_geometry(coord_frame)

        # Set initial view
        for name in view_names:
            ctr = visualizers[name].get_view_control()
            ctr.set_zoom(0.5)
            ctr.set_front([0, -1, -0.5])
            ctr.set_lookat([0.4, 0, 0.1])
            ctr.set_up([0, 0, 1])

        last_update_time = time.time()

        try:
            while self.running:
                current_time = time.time()

                # Update frame if not paused and in auto-play mode
                if not self.frame_mode and not self.paused:
                    if current_time - last_update_time >= self.frame_interval:
                        self.current_frame = (self.current_frame + 1) % self.num_frames
                        last_update_time = current_time

                # Get geometries for current frame
                geometries = self._create_visualization_geometry(self.current_frame)

                # Update point clouds
                for name, pcd in geometries:
                    if name in point_clouds:
                        point_clouds[name].points = pcd.points
                        point_clouds[name].colors = pcd.colors
                        visualizers[name].update_geometry(point_clouds[name])

                # Poll events and render
                for name in view_names:
                    visualizers[name].poll_events()
                    visualizers[name].update_renderer()

                # Check if any window was closed
                for name in view_names:
                    if not visualizers[name].poll_events():
                        self.running = False
                        break

                # Print frame info
                print(f"\rFrame: {self.current_frame + 1}/{self.num_frames} "
                      f"{'[PAUSED]' if self.paused else '[PLAYING]'}", end='', flush=True)

                time.sleep(0.01)  # Small sleep to prevent CPU overload

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            for vis in visualizers.values():
                vis.destroy_window()
            print("\nVisualization ended.")

    def _quit_callback(self):
        self.running = False
        return True

    def _toggle_pause(self):
        self.paused = not self.paused
        return True

    def _next_frame(self):
        self.current_frame = (self.current_frame + 1) % self.num_frames
        return True

    def _prev_frame(self):
        self.current_frame = (self.current_frame - 1) % self.num_frames
        return True

    def _reset_frame(self):
        self.current_frame = 0
        return True


def main(
    episode_idx=0,
    # Femto settings
    enable_femto_transform=True,
    enable_femto_cropping=True,
    enable_femto_variance_filter=True,
    femto_variance_kernel_size=5,
    femto_variance_threshold=50.0,
    # D405 settings
    enable_wrist_camera=True,
    enable_d405_transform=True,
    enable_d405_variance_filter=True,
    d405_variance_kernel_size=5,
    d405_variance_threshold=50.0,
    # Visualization flags
    show_femto=True,
    show_d405=True,
    show_fusion=True,
    # Playback
    frame_mode=False,
    playback_fps=10,
):
    visualizer = DatasetVisualizer(
        dataset_path=path_dataset,
        episode_idx=episode_idx,
        enable_femto_transform=enable_femto_transform,
        enable_femto_cropping=enable_femto_cropping,
        enable_femto_variance_filter=enable_femto_variance_filter,
        femto_variance_kernel_size=femto_variance_kernel_size,
        femto_variance_threshold=femto_variance_threshold,
        enable_wrist_camera=enable_wrist_camera,
        enable_d405_transform=enable_d405_transform,
        enable_d405_variance_filter=enable_d405_variance_filter,
        d405_variance_kernel_size=d405_variance_kernel_size,
        d405_variance_threshold=d405_variance_threshold,
        show_femto=show_femto,
        show_d405=show_d405,
        show_fusion=show_fusion,
        frame_mode=frame_mode,
        playback_fps=playback_fps,
    )
    visualizer.run()


if __name__ == '__main__':
    main()
