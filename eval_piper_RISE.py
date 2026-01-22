# eval_piper_RISE.py
import time
from multiprocessing.managers import SharedMemoryManager
import click
import open3d as o3d
import os
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
import MinkowskiEngine as ME
from termcolor import cprint

from pcdp.real_world.real_env_piper import RealEnv
from pcdp.real_world.teleoperation_piper import TeleoperationPiper
from pcdp.common.precise_sleep import precise_wait
import pcdp.common.mono_time as mono_time
from pcdp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from pcdp.policy.diffusion_RISE_policy import RISEPolicy
from pcdp.common.RISE_transformation import xyz_rot_transform
from pcdp.dataset.RISE_util import *
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor, LowDimPreprocessor
from pcdp.model.common.normalizer import LinearNormalizer


robot_to_base = np.array([
    [1.,         0.,         0.,          -0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])


# Hydra 설정 등록
OmegaConf.register_new_resolver("eval", eval, replace=True)


import numpy as np
from pcdp.common import RISE_transformation as rise_tf


def revert_action_transformation(transformed_action_6d, robot_to_base_matrix):
    
    base_to_robot_matrix = np.linalg.inv(robot_to_base_matrix)
    original_poses = []
    for action in transformed_action_6d:
        translation = action[:3]
        rotation = action[3:6]
        transformed_matrix = rise_tf.rot_trans_mat(translation, rotation)

        # 순방향: T_k = robot_to_base @ eef_to_robot_base_k
        # 역방향: eef_to_robot_base_k = inv(robot_to_base) @ T_k
        original_matrix = base_to_robot_matrix @ transformed_matrix

        # 5. 원래의 4x4 행렬을 다시 6D pose [x,y,z,r,p,y] 벡터로 변환합니다.
        original_pose_6d = rise_tf.mat_to_xyz_rot(
            original_matrix,
            rotation_rep='euler_angles',
            rotation_rep_convention='ZYX'
        )
        original_poses.append(original_pose_6d)
    return np.array(original_poses, dtype=np.float32)


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save evaluation results.')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode for initial condition visualization')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--save-data', is_flag=True, default=False, help="Enable saving episode data (pointcloud, robot state).")
def main(input, output, match_episode, frequency, save_data):
    # 체크포인트 및 설정 로드
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # 정책 모델 초기화 및 가중치 로드
    policy: RISEPolicy = hydra.utils.instantiate(cfg.policy)
    
    # EMA 가중치가 있으면 사용하고, 없으면 일반 모델 가중치를 사용
    state_dict = payload['state_dicts']['model']
    
    policy.load_state_dict(state_dict)
    
    device = torch.device(cfg.training.device)
    policy.to(device).eval()
    cprint(f"Policy loaded on {device}", "green")
    
    # =========================
    # Normalizer 설정 (주석 처리됨)
    # 모델 체크포인트에 정규화 정보가 이미 포함되어 있으므로, 별도로 로드하거나 계산할 필요가 없습니다.
    # policy.load_state_dict()를 통해 모델 가중치를 로드할 때 정규화 파라미터도 함께 로드됩니다.
    # =========================
    # ckpt_dir = os.path.dirname(ckpt_path)
    # n_path = os.path.join(ckpt_dir, "normalizer.pt")
    # if os.path.exists(n_path):
    #     state = torch.load(n_path, map_location=device)
    #     n = LinearNormalizer(); n.load_state_dict(state)
    #     policy.set_normalizer(n)
    #     cprint(f"Loaded normalizer: {n_path}", "green")
    # else:
    #     # ⬇️ 학습과 같은 translation 정규화 범위를 강제 (오프셋 방지)
    #     cfg.task.dataset._target_ = 'pcdp.dataset.RISE_stack_dataset.RISE_RealStackPointCloudDataset'
    #     dataset = hydra.utils.instantiate(cfg.task.dataset)
    #     if 'translation' in cfg.training:
    #         dataset.set_translation_norm_config(cfg.training.translation)
    #     normalizer = dataset.get_normalizer(device=device)
    #     policy.set_normalizer(normalizer)
    #     cprint("Built normalizer from dataset with training.translation enforced.", "green")

    # (선택) 정규화기 중심 확인: 정규화 공간의 0이 물리공간 어디인지
    try:
        pd = policy.normalizer['action_translation'].params_dict  # SingleFieldLinearNormalizer
        center = (-pd['offset'] / pd['scale']).detach().cpu().numpy()
        cprint(f"action_translation center (m): {center}", "cyan")
    except Exception as e:
        cprint(f"[warn] center print failed: {e}", "yellow")

    dt = 1.0 / frequency
    voxel_size = cfg.task.dataset.voxel_size

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            TeleoperationPiper(shm_manager=shm_manager) as teleop, \
            RealEnv(
                output_dir=output,
                frequency=frequency,
                init_joints=True,
                orbbec_mode="C2D",
                shm_manager=shm_manager,
                save_data=save_data
            ) as env:
            
            cprint('Ready! Press "C" to start evaluation, "S" to stop, "Q" to quit.', "yellow")
            
            # YAML 설정에서 preprocessor_config를 가져와 PointCloudPreprocessor 인스턴스화
            pc_preprocessor = PointCloudPreprocessor(**cfg.task.dataset.pc_preprocessor_config)
            
            target_pose = [0.054952, 0.0, 0.493991, 0.0, np.deg2rad(85.0), 0.0, 0.0]
            plan_time = mono_time.now_s() + 2.0
            env.exec_actions([target_pose], [plan_time])
            time.sleep(2.0)
            
            t_start = mono_time.now_s()
            iter_idx = 0
            is_evaluating = False
            cnt=0

            test_start = 0
            t2 = 0
            

            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt * 1
                t0 = mono_time.now_ms()
                obs = env.get_obs()
                # 키보드 입력 처리
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        if is_evaluating:
                            env.end_episode()
                        return
                    elif key_stroke == KeyCode(char='c'):
                        if not is_evaluating:
                            env.start_episode()
                            is_evaluating = True
                            test_start= mono_time.now_ms()
                            cprint("Evaluation started!", "cyan")
                    elif key_stroke == KeyCode(char='s'):
                        if is_evaluating:
                            env.end_episode()
                            is_evaluating = False
                            cprint("Evaluation stopped. Human in control.", "yellow")
                
                
                if is_evaluating:
                    with torch.no_grad():
                        # 1. 관측 데이터 전처리 (학습 파이프라인과 일치시킴)
                        pc_raw = obs['main_pointcloud'][-1]

                        
                        # 포인트클라우드 전처리
                        pc = pc_preprocessor.process(pc_raw)
                        coords = np.floor(pc[:, :3] / voxel_size).astype(np.int32)
                        coords = np.ascontiguousarray(coords)
                        feats = pc.astype(np.float32)
                        coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
                        cloud_data = ME.SparseTensor(
                            features=feats_batch, 
                            coordinates=coords_batch,
                            device=device)
                        
                        t1 = mono_time.now_ms()
                        if cnt%10 ==0:
                            pred_action_10d = policy(cloud_data, batch_size=1)

                            print(f"inference: {mono_time.now_ms() - t1}")
                            print(f"loop time: {mono_time.now_ms() - t2}")
                            t2 = mono_time.now_ms()

                            # 3. 액션 후처리
                            pred = pred_action_10d
                            if pred.ndim == 3:
                                pred = pred.squeeze(0)

                            pos   = pred[:, :3].cpu().numpy()
                            rot6d = pred[:, 3:9].cpu().numpy()
                            grip = np.where(pred[:, 9:].cpu().numpy() > 55, 85, 0).astype(np.int32) 
                            xyz_rot6d = np.concatenate([pos, rot6d], axis=-1)

                            # 6D 회전 표현을 오일러 각도로 변환
                            xyz_euler_base_frame = xyz_rot_transform(
                                xyz_rot6d,
                                from_rep="rotation_6d",
                                to_rep="euler_angles",
                                to_convention="ZYX"
                            )

                            # 액션을 "base" 좌표계에서 로봇의 실제 실행 좌표계로 역변환
                            xyz_euler_robot_frame = revert_action_transformation(xyz_euler_base_frame, robot_to_base)

                            action_sequence_7d = np.concatenate([xyz_euler_robot_frame, grip], axis=-1)

                            # 4. 로봇 제어
                            obs_timestamps = obs['timestamp']          # mono_time 기반이어야 함
                            L = len(action_sequence_7d)
                            action_offset = 0
                            action_exec_latency = 0.02 # 100ms

                            action_timestamps = (np.arange(L, dtype=np.float64) + action_offset) * dt + obs_timestamps[-1]
                            is_new = action_timestamps > (mono_time.now_s() + action_exec_latency)


                            if not np.any(is_new):
                                # 모두 과거면: 다음 슬롯에 마지막 1스텝만 예약
                                next_step_time = mono_time.now_s() + dt
                                action_timestamps = np.array([next_step_time], dtype=np.float64)
                                action_sequence_7d = action_sequence_7d[[-1]]
                            else:
                                action_timestamps = action_timestamps[is_new]
                                action_sequence_7d = action_sequence_7d[is_new]



                            env.exec_actions(
                                actions=action_sequence_7d,
                                timestamps=action_timestamps
                            )
                        if mono_time.now_ms()- test_start > 60000:
                            env.end_episode()
                            is_evaluating=False
                        cnt+=1
                else:
                    # ===== 사람 제어 (Human Control) =====
                    target_pose = teleop.get_motion_state()
                    env.exec_actions(actions=[target_pose], timestamps=[mono_time.now_s() + dt])
                # print(f"loop time: {mono_time.now_ms() - t0}")
                precise_wait(t_cycle_end)
                iter_idx += 1

if __name__ == '__main__':
    main()