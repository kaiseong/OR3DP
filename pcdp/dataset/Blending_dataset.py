#!/usr/bin/env python3
"""
Blending Dataset for training
Loads raw images from Zarr and applies preprocessing on-the-fly
"""

import numpy as np
import zarr
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset

from pcdp.real_world.blending_processor import BlendingProcessor


class BlendingDataset(Dataset):
    """
    Dataset for point cloud blending research

    Loads raw color/depth images and applies preprocessing during training:
    - Depth image to point cloud conversion
    - Threshold filtering (configurable)
    - Spatial filtering (optional)
    - Blending
    - Voxel downsampling (optional)
    - Workspace cropping
    """

    def __init__(
        self,
        dataset_dir: str,
        episode_ids: Optional[List[int]] = None,
        # Preprocessing parameters
        apply_threshold_filter: bool = True,
        threshold_min_dist: float = 0.09,
        threshold_max_dist: float = 0.20,
        # Point cloud processing
        voxel_size: Optional[float] = None,  # mm, None = no voxelization
        max_points: Optional[int] = None,    # None = no downsampling
        workspace_bounds: Optional[np.ndarray] = None,  # (3, 2) [x, y, z] ranges in mm
        # Output options
        return_raw: bool = False,  # If True, also return raw images
        transform=None,
    ):
        """
        Args:
            dataset_dir: Path to dataset root (contains episode folders)
            episode_ids: List of episode IDs to load, None = all episodes
            apply_threshold_filter: Apply depth threshold filter to D405
            threshold_min_dist: Min distance for D405 (meter)
            threshold_max_dist: Max distance for D405 (meter)
            voxel_size: Voxel size for downsampling (mm), None = no voxelization
            max_points: Maximum number of points, None = no downsampling
            workspace_bounds: Workspace bounds in mm, shape (3, 2) [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            return_raw: If True, return raw images in addition to point cloud
            transform: Optional transform to apply to point cloud
        """
        self.dataset_dir = Path(dataset_dir)
        self.apply_threshold_filter = apply_threshold_filter
        self.threshold_min_dist = threshold_min_dist
        self.threshold_max_dist = threshold_max_dist
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.workspace_bounds = workspace_bounds
        self.return_raw = return_raw
        self.transform = transform

        # Load intrinsics (shared across all episodes)
        intrinsics_file = self.dataset_dir / "intrinsics.npz"
        if not intrinsics_file.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_file}")

        intrinsics = np.load(intrinsics_file)
        self.femto_intrinsics = intrinsics['femto_intrinsics']
        self.d405_intrinsics = intrinsics['d405_intrinsics']

        # Initialize BlendingProcessor (without cameras)
        self.processor = BlendingProcessor(
            femto_camera=None,
            d405_camera=None,
            piper_interface=None
        )

        # Build episode index
        self.episodes = []
        if episode_ids is None:
            # Load all episodes
            episode_dirs = sorted(self.dataset_dir.glob("episode_*"))
        else:
            episode_dirs = [self.dataset_dir / f"episode_{i:06d}" for i in episode_ids]

        for ep_dir in episode_dirs:
            if not ep_dir.exists():
                continue

            zarr_path = ep_dir / "data.zarr"
            if not zarr_path.exists():
                continue

            root = zarr.open(str(zarr_path), mode='r')
            num_frames = root['femto_color'].shape[0]

            # Store episode info
            for frame_idx in range(num_frames):
                self.episodes.append({
                    'episode_dir': ep_dir,
                    'zarr_path': zarr_path,
                    'frame_idx': frame_idx,
                    'episode_id': int(ep_dir.name.split('_')[-1])
                })

        print(f"[BlendingDataset] Loaded {len(self.episodes)} frames from {len(episode_dirs)} episodes")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'pointcloud': (N, 6) tensor [x, y, z, r, g, b], Femto camera frame, mm, RGB 0-1
                'robot_eef_pose': (6,) tensor [x, y, z, rx, ry, rz], meter, radians
                'timestamp': float
                'stage': int
                'episode_id': int
                'frame_idx': int

                # Optional (if return_raw=True):
                'femto_color': (720, 1280, 3) uint8
                'femto_depth': (288, 320) uint16
                'd405_color': (240, 424, 3) uint8
                'd405_depth': (240, 424) uint16
            }
        """
        episode_info = self.episodes[idx]

        # Load zarr data
        root = zarr.open(str(episode_info['zarr_path']), mode='r')
        frame_idx = episode_info['frame_idx']

        femto_color = root['femto_color'][frame_idx]
        femto_depth = root['femto_depth'][frame_idx]
        d405_color = root['d405_color'][frame_idx]
        d405_depth = root['d405_depth'][frame_idx]
        robot_eef_pose = root['robot_eef_pose'][frame_idx]
        timestamp = root['timestamp'][frame_idx]
        stage = root['stage'][frame_idx]

        # Process raw images to blended point cloud
        blended_pc = self.processor.process_from_images(
            femto_color=femto_color,
            femto_depth=femto_depth,
            femto_intrinsics=self.femto_intrinsics,
            d405_color=d405_color,
            d405_depth=d405_depth,
            d405_intrinsics=self.d405_intrinsics,
            ee_pose=robot_eef_pose,
            apply_threshold_filter=self.apply_threshold_filter,
            threshold_min_dist=self.threshold_min_dist,
            threshold_max_dist=self.threshold_max_dist
        )

        # Apply workspace bounds if specified
        if self.workspace_bounds is not None:
            mask = (
                (blended_pc[:, 0] >= self.workspace_bounds[0, 0]) &
                (blended_pc[:, 0] <= self.workspace_bounds[0, 1]) &
                (blended_pc[:, 1] >= self.workspace_bounds[1, 0]) &
                (blended_pc[:, 1] <= self.workspace_bounds[1, 1]) &
                (blended_pc[:, 2] >= self.workspace_bounds[2, 0]) &
                (blended_pc[:, 2] <= self.workspace_bounds[2, 1])
            )
            blended_pc = blended_pc[mask]

        # Voxel downsampling if specified
        if self.voxel_size is not None:
            blended_pc = self._voxel_downsample(blended_pc, self.voxel_size)

        # Random downsampling if specified
        if self.max_points is not None and blended_pc.shape[0] > self.max_points:
            indices = np.random.choice(blended_pc.shape[0], self.max_points, replace=False)
            blended_pc = blended_pc[indices]

        # Apply transform if specified
        if self.transform is not None:
            blended_pc = self.transform(blended_pc)

        # Convert to torch tensors
        result = {
            'pointcloud': torch.from_numpy(blended_pc).float(),
            'robot_eef_pose': torch.from_numpy(robot_eef_pose).float(),
            'timestamp': float(timestamp),
            'stage': int(stage),
            'episode_id': episode_info['episode_id'],
            'frame_idx': episode_info['frame_idx']
        }

        # Optionally return raw images
        if self.return_raw:
            result.update({
                'femto_color': torch.from_numpy(femto_color),
                'femto_depth': torch.from_numpy(femto_depth),
                'd405_color': torch.from_numpy(d405_color),
                'd405_depth': torch.from_numpy(d405_depth)
            })

        return result

    def _voxel_downsample(self, pc: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Voxel downsampling

        Args:
            pc: (N, 6) [x, y, z, r, g, b]
            voxel_size: Voxel size in mm

        Returns:
            downsampled_pc: (M, 6)
        """
        if pc.shape[0] == 0:
            return pc

        xyz = pc[:, :3]
        rgb = pc[:, 3:]

        # Compute voxel indices
        voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)

        # Find unique voxels
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

        # Average points within each voxel (simple: just take first point)
        # For better quality, you can average XYZ and RGB
        downsampled_pc = pc[unique_indices]

        return downsampled_pc

    def get_episode_length(self, episode_id: int) -> int:
        """Get number of frames in an episode"""
        episode_dir = self.dataset_dir / f"episode_{episode_id:06d}"
        zarr_path = episode_dir / "data.zarr"
        root = zarr.open(str(zarr_path), mode='r')
        return root['femto_color'].shape[0]

    def get_episode_frames(self, episode_id: int) -> List[int]:
        """Get list of frame indices for an episode in the dataset"""
        return [i for i, ep in enumerate(self.episodes) if ep['episode_id'] == episode_id]


# Example usage
if __name__ == '__main__':
    # Define workspace bounds (in mm, Femto camera frame)
    workspace_bounds = np.array([
        [0.0, 740.0],      # X range
        [-400.0, 350.0],   # Y range
        [-100.0, 600.0]    # Z range
    ])

    # Create dataset
    dataset = BlendingDataset(
        dataset_dir="/home/leejungwook/pcdp_backup/data/blending_dataset",
        episode_ids=None,  # Load all episodes
        apply_threshold_filter=True,
        threshold_min_dist=0.04,
        threshold_max_dist=0.50,
        voxel_size=5.0,  # 5mm voxel
        max_points=8192,  # Downsample to 8K points
        workspace_bounds=workspace_bounds,
        return_raw=False
    )

    # Test loading
    print(f"\nDataset size: {len(dataset)} frames")

    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Pointcloud shape: {sample['pointcloud'].shape}")
    print(f"Robot pose: {sample['robot_eef_pose']}")
    print(f"Episode ID: {sample['episode_id']}, Frame: {sample['frame_idx']}")

    # Create DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    print(f"\nDataLoader batch shape:")
    for batch in dataloader:
        print(f"  Pointcloud: {batch['pointcloud'].shape}")
        print(f"  Robot pose: {batch['robot_eef_pose'].shape}")
        break
