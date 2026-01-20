"""
RISE Blending Data Collection - Using RealEnv (PCDP-compatible)

This script uses RealEnv with camera_mode="blending" to ensure:
1. Proper timestamp synchronization (same as PCDP)
2. Automatic data recording via Recorder
3. Fair comparison for paper experiments (only difference is wrist camera)
"""

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
from termcolor import cprint
from pcdp.real_world.real_env_piper import RealEnv
from pcdp.real_world.teleoperation_piper import TeleoperationPiper
from pcdp.common.precise_sleep import precise_wait
import pcdp.common.mono_time as mono_time
from pcdp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

def make_status_panel(episode_id: int, recording: bool, stage: int, extra_text: str=""):
    h, w = 120, 520
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    dot_color = (0, 200, 0) if recording else(0, 0, 200)
    cv2.circle(panel, (20, 30), 10, dot_color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(panel, "Recording" if recording else "Stopped",
                (40, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

    cv2.putText(panel, f"Episode: {episode_id}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(panel, f"Stage:  {stage}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    if extra_text:
        cv2.putText(panel, extra_text, (260, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return panel


@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--save-data', is_flag=True, default=True, help="Enable saving episode data (pointcloud, robot state).")
def main(output, init_joints, frequency, command_latency, save_data):
    dt = 1/frequency

    # IK parameters
    urdf_path = "/home/moai/wrist_rise/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
    mesh_dir = "/home/moai/wrist_rise/dependencies"
    ee_link_name = "link6"
    joints_to_lock_names = []

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            TeleoperationPiper(shm_manager=shm_manager) as ms, \
            RealEnv(
                output_dir=output,
                # IK params
                urdf_path=urdf_path,
                mesh_dir=mesh_dir,
                ee_link_name=ee_link_name,
                joints_to_lock_names=joints_to_lock_names,
                # recording resolution
                frequency=frequency,
                n_obs_steps=1,
                init_joints=init_joints,
                orbbec_mode="C2D",
                camera_mode="blending",
                shm_manager=shm_manager,
                save_data=save_data
            ) as env:
            cv2.setNumThreads(1)

            base_pose = [0.054952, 0.0, 0.493991, 0.0, np.deg2rad(85.0), 0.0, 0.0]
            plan_time = mono_time.now_s() + 2.0
            env.exec_actions([base_pose], [plan_time])
            cprint("Moving to the base_pose, please wait...", "yellow")

            time.sleep(3.0)
            cprint('Ready!', "green", attrs=["bold"])

            state = env.get_robot_state()
            target_pose = base_pose.copy()
            t_start = mono_time.now_s()
            iter_idx = 0
            stop = False
            is_recording = False

            stage = 0

            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                obs = env.get_obs()
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        env.start_episode(mono_time.now_s()+2*dt)
                        key_counter.clear()
                        is_recording = True
                        cprint('Recording!', "green", attrs=["bold"])
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        cprint('Stopped.', "yellow")
                    elif key_stroke == Key.backspace:
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                    elif isinstance(key_stroke, KeyCode) and key_stroke.char is not None and key_stroke.char.isdigit():
                        stage = int(key_stroke.char)
                        cprint(f"[Stage] set to {stage}", "cyan")

                episode_id = env.recorder.n_episodes
                panel = make_status_panel(
                    episode_id=episode_id,
                    recording=is_recording,
                    stage=stage,
                    extra_text="C: Start  S: stop  Q: quit 0-9: stage"
                )
                cv2.imshow("Status", panel)
                cv2.waitKey(1)

                precise_wait(t_sample)
                target_pose = ms.get_motion_state()
                action_to_record = obs['robot_eef_pose'][-1]
                env.exec_actions(
                    actions=[target_pose],
                    timestamps=[t_command_target],
                    stages=[stage],
                    recorded_actions=[action_to_record])
                precise_wait(t_cycle_end)
                iter_idx += 1

            try:
                cv2.destroyWindow("Status")
            except cv2.error:
                pass

            cprint("\nShutting down...", "yellow")

if __name__ == '__main__':
    main()
