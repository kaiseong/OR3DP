# PCM_stack_dataset.py
from typing import Dict, List
import torch
import numpy as np
import zarr
import os
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import json
import hashlib
import copy
import MinkowskiEngine as ME
import collections.abc as container_abcs
import torch.nn as nn

from omegaconf import OmegaConf, DictConfig, ListConfig
from pcdp.common.pytorch_util import dict_apply
from pcdp.dataset.base_dataset import BasePointCloudDataset
from pcdp.model.common.normalizer import LinearNormalizer
from pcdp.common.replay_buffer import ReplayBuffer
from pcdp.real_world.real_data_pc_conversion import (
    _get_replay_buffer,
    PointCloudPreprocessor,
    LowDimPreprocessor,
    FemtoConfig,
    D405Config,
    robot_to_base as DEFAULT_ROBOT_TO_BASE,
)
from pcdp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from pcdp.model.common.normalizer import SingleFieldLinearNormalizer
from pcdp.common.normalize_util import get_norm_stats_in_batch

from tqdm import tqdm
from pcdp.common.RISE_transformation import xyz_rot_transform


def _to_py(x):
    # DictConfig/ListConfig → 순수 파이썬 컨테이너
    if isinstance(x, (DictConfig, ListConfig)):
        return OmegaConf.to_container(x, resolve=True)
    return x


def _build_pc_preprocessor_from_cfg(pc_preprocessor_config):

    if pc_preprocessor_config is None:
        return None

    cfg = _to_py(pc_preprocessor_config)
    cfg = dict(cfg)  # shallow copy

    if 'workspace_bounds' not in cfg:
        raise ValueError("pc_preprocessor_config에 workspace_bounds가 필요합니다.")
    workspace_bounds = np.asarray(cfg.pop('workspace_bounds'), dtype=np.float64)

    sensor_mode = cfg.pop('sensor_mode', 'both')

    femto_cfg_data = cfg.pop('femto', None)

    if femto_cfg_data is None and all(k in cfg for k in ('extrinsics_matrix', 'K_depth', 'depth_width', 'depth_height')):
        femto_cfg_data = {
            'cam_to_base': cfg.pop('extrinsics_matrix'),
            'K': cfg.pop('K_depth'),
            'width': cfg.pop('depth_width'),
            'height': cfg.pop('depth_height'),
            'var_ksize': cfg.pop('var_ksize', 5),
            'var_std_thresh_m': cfg.pop('var_std_thresh_m', 0.004),
            'var_min_depth_m': cfg.pop('var_min_depth_m', 0.01),
            'var_max_depth_m': cfg.pop('var_max_depth_m', 1e6),
        }

    femto_cfg = FemtoConfig(**femto_cfg_data) if femto_cfg_data is not None else None

    # D405 config
    d405_cfg_data = cfg.pop('d405', None)
    if d405_cfg_data is not None:
        d405_cfg_data = dict(d405_cfg_data)
        d405_cfg_data.setdefault('robot_to_base', DEFAULT_ROBOT_TO_BASE)
        d405_cfg = D405Config(**d405_cfg_data)
    else:
        d405_cfg = None

    if sensor_mode in ('both', 'd405') and d405_cfg is None:
        if femto_cfg is not None:
            sensor_mode = 'femto'  
        else:
            raise ValueError("sensor_mode에 d405가 포함되어 있지만 d405 설정이 없습니다.")
    if sensor_mode in ('both', 'femto') and femto_cfg is None:
        if d405_cfg is not None:
            sensor_mode = 'd405'
        else:
            raise ValueError("sensor_mode에 femto가 포함되어 있지만 femto 설정이 없습니다.")

    return PointCloudPreprocessor(
        workspace_bounds=workspace_bounds,
        femto=femto_cfg,
        d405=d405_cfg,
        sensor_mode=sensor_mode,
        **cfg,
    )

class PCM_RealStackPointCloudDataset(BasePointCloudDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=20,
            pad_before=0,
            pad_after=0,
            n_obs_steps=1,
            n_latency_steps=0,
            use_cache=True,
            seed=42,
            voxel_size=0.005,
            val_ratio=0.0,
            max_train_episodes=None,
            enable_pc_preprocessing=True,
            pc_preprocessor_config=None,
            enable_low_dim_preprocessing=True,
            low_dim_preprocessor_config=None,
            downsample_factor=3,
            downsample_use_all_offsets: bool = False,
            group_by_offsets = None,

        ):

        super().__init__() 

        assert os.path.isdir(dataset_path), f"Dataset path does not exist: {dataset_path}"

        pc_preprocessor = None
        if enable_pc_preprocessing:
            pc_preprocessor = _build_pc_preprocessor_from_cfg(pc_preprocessor_config)
        
        low_dim_preprocessor = None
        if enable_low_dim_preprocessing:
            ld_kwargs = OmegaConf.to_container(low_dim_preprocessor_config, resolve=True) if low_dim_preprocessor_config else {}
            low_dim_preprocessor = LowDimPreprocessor(**ld_kwargs)
        
        replay_buffer = None
        if use_cache:
            shape_meta_py = _to_py(shape_meta)
            pc_conf_py = _to_py(pc_preprocessor_config) if 'pc_preprocessor_config' in locals() else None
            ld_conf_py = _to_py(low_dim_preprocessor_config) if 'low_dim_preprocessor_config' in locals() else None

            cache_key = {
                "shape_meta": shape_meta_py,
                "downsample_factor": int(downsample_factor),
                "downsample_use_all_offsets": bool(downsample_use_all_offsets),
                "pc_preprocessor_config": pc_conf_py,
                "lowdim_preprocessor_config": ld_conf_py,
            }

            cache_json = json.dumps(cache_key, sort_keys=True, ensure_ascii=False)
            # cache_hash = hashlib.md5(cache_json.encode("utf-8")).hexdigest()
            # cache_zarr_path = os.path.join(dataset_path, cache_hash + ".zarr.zip")
            
            # cache_json = json.dumps(cache_key, sort_keys=True)
            cache_hash = hashlib.md5(cache_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, cache_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    print('Cache does not exist. Creating!')
                    replay_buffer = _get_replay_buffer(
                        dataset_path=dataset_path,
                        shape_meta=shape_meta,
                        store=zarr.MemoryStore(),
                        pc_preprocessor=pc_preprocessor,
                        lowdim_preprocessor=low_dim_preprocessor,
                        downsample_factor=downsample_factor,
                        downsample_use_all_offsets=downsample_use_all_offsets
                    )
                    print('Saving cache to disk.')
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
                pc_preprocessor=pc_preprocessor,
                lowdim_preprocessor=low_dim_preprocessor,
                downsample_factor=downsample_factor,
                downsample_use_all_offsets=downsample_use_all_offsets
            )
        
        pointcloud_keys = [k for k, v in shape_meta.obs.items() if v.type == 'pointcloud']
        lowdim_keys = [k for k, v in shape_meta.obs.items() if v.type == 'low_dim']
        
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in pointcloud_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
        
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        if group_by_offsets and downsample_use_all_offsets and downsample_factor > 1:
            g = downsample_factor
            n_groups = replay_buffer.n_episodes // g
            group_val = get_val_mask(n_groups, val_ratio=val_ratio, seed=seed)
            val_mask = np.repeat(group_val, g)[:replay_buffer.n_episodes]
        train_mask = ~val_mask

        if max_train_episodes is not None:
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)
        
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.pointcloud_keys = pointcloud_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask
        self.horizon = horizon
        self.voxel_size = voxel_size
        self.n_latency_steps = n_latency_steps
        self.normalizer = None
        self.translation_norm_config = None

    def set_translation_norm_config(self, config):
        self.translation_norm_config = config

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler =SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        val_set.val_mask = self.val_mask
        val_set.normalizer = self.normalizer
        return val_set
    def __len__(self):
        return len(self.sampler)
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_normalizer(self, device='cpu') -> LinearNormalizer:
        normalizer = LinearNormalizer()

        if self.translation_norm_config is not None:
            # Use pre-defined min/max for translation
            translation_config = np.array(self.translation_norm_config)
            t_min = torch.from_numpy(translation_config[:, 0]).to(dtype=torch.float32, device=device)
            t_max = torch.from_numpy(translation_config[:, 1]).to(dtype=torch.float32, device=device)

            # Create normalizer from min/max
            # Logic from _fit function in normalizer.py
            output_max = 1.0
            output_min = -1.0
            range_eps = 1e-4
            
            input_range = t_max - t_min
            input_range[input_range < range_eps] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * t_min
            
            # Create ParameterDict for the normalizer
            params_dict = nn.ParameterDict({
                'scale': scale,
                'offset': offset,
                'input_stats': nn.ParameterDict({
                    'min': t_min, 'max': t_max,
                    'mean': torch.zeros_like(t_min), 'std': torch.ones_like(t_min)
                })
            })
            for p in params_dict.parameters():
                p.requires_grad_(False)

            # Apply the same normalizer to both obs and action translation
            translation_normalizer = SingleFieldLinearNormalizer(params_dict)
            normalizer['action_translation'] = translation_normalizer
            normalizer['obs_translation'] = translation_normalizer
        else:
            # Fallback to data-driven normalization for translation
            all_actions = torch.from_numpy(self.replay_buffer['action'][:]).to(device)
            all_robot_obs = torch.from_numpy(self.replay_buffer['robot_obs'][:]).to(device)
            action_trans_data = all_actions[:, :3]
            obs_trans_data = all_robot_obs[:, :3]
            normalizer['action_translation'] = SingleFieldLinearNormalizer.create_fit(action_trans_data)
            normalizer['obs_translation'] = SingleFieldLinearNormalizer.create_fit(obs_trans_data)

        # Gripper and color normalization remains data-driven
        all_robot_obs = torch.from_numpy(self.replay_buffer['robot_obs'][:]).to(device)
        obs_grip_data = all_robot_obs[:, 6:7]
        normalizer['obs_gripper'] = SingleFieldLinearNormalizer.create_fit(obs_grip_data)

        all_actions = torch.from_numpy(self.replay_buffer['action'][:]).to(device)
        action_grip_data = all_actions[:, 6:7]
        normalizer['action_gripper'] = SingleFieldLinearNormalizer.create_fit(action_grip_data)


        # ========= MODIFIED: Memory-Efficient Normalization =========
        # New code to calculate stats in batches, avoiding memory overflow.
        color_stats = get_norm_stats_in_batch(
            replay_buffer=self.replay_buffer,
            key='pointcloud',
            obs_slice=slice(3, 6),
            device=device
        )
        
        if color_stats is not None:
            # Create normalizer from calculated stats
            color_stats_torch = {k: torch.from_numpy(v).to(device) for k, v in color_stats.items()}
            # Add min/max for compatibility if they don't exist
            if 'min' not in color_stats_torch:
                color_stats_torch['min'] = torch.full_like(color_stats_torch['mean'], -1.0)
            if 'max' not in color_stats_torch:
                color_stats_torch['max'] = torch.full_like(color_stats_torch['mean'], 1.0)

            color_normalizer = SingleFieldLinearNormalizer.create_manual(
                scale=1.0 / (color_stats_torch['std'] + 1e-6),
                offset=-color_stats_torch['mean'] / (color_stats_torch['std'] + 1e-6),
                input_stats_dict=color_stats_torch
            )
            normalizer['pointcloud_color'] = color_normalizer

        return normalizer

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)
        obs_dict = {key: data[key][T_slice] for key in self.pointcloud_keys + self.lowdim_keys}
        clouds = [obs_dict['pointcloud'][i].astype(np.float32) for i in range(self.n_obs_steps)]
        robot_obs_euler = np.array([obs_dict['robot_obs'][i] for i in range(self.n_obs_steps)], dtype=np.float32)
        
        actions_euler = data['action'].astype(np.float32)

        # Convert obs from Euler angles to 6D representation
        obs_pose_euler = robot_obs_euler[:, :6]
        obs_gripper = robot_obs_euler[:, 6:]
        obs_9d = xyz_rot_transform(obs_pose_euler, from_rep='euler_angles', to_rep='rotation_6d', from_convention='ZYX')
        robot_obs_10d = np.concatenate([obs_9d, obs_gripper], axis=-1)

        # Convert action from Euler angles to 6D representation
        pose_euler = actions_euler[:, :6]
        gripper_action = actions_euler[:, 6:]
        pose_9d = xyz_rot_transform(pose_euler, from_rep="euler_angles", to_rep="rotation_6d", from_convention="ZYX")
        actions_10d = np.concatenate([pose_9d, gripper_action], axis=-1)

        # Voxelize for MinkowskiEngine
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            coords = np.floor(cloud[:, :3] / self.voxel_size).astype(np.int32)
            coords = np.ascontiguousarray(coords)
            
            # Return unnormalized RGB in features
            input_feats_list.append(cloud.astype(np.float32))
            input_coords_list.append(coords)

        return {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'robot_obs': torch.from_numpy(robot_obs_10d).float(),
            'action': torch.from_numpy(actions_10d).float(),
        }

def collate_fn(batch):
    if not isinstance(batch, list):
        return batch
    
    elem = batch[0]
    if not isinstance(elem, container_abcs.Mapping):
        raise TypeError(f"Batch must contain dicts, but found {type(elem)}")
    
    ret_dict = {}
    coords_list = list()
    feats_list = list()

    for key in elem:
        if key in ['input_coords_list', 'input_feats_list']:
            flat_list = [item for d in batch for item in d[key]]
            if key == 'input_coords_list':
                coords_list.extend(flat_list)
            else:
                feats_list.extend(flat_list)
        else:
            ret_dict[key] = torch.stack([d[key] for d in batch], dim =0)
    if coords_list and feats_list:
        coords_batch, feats_batch = ME.utils.sparse_collate(coords=coords_list, feats=feats_list)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
    return ret_dict
