"""
Usage:
(robodiff)$ python demo_real_piper.py -o <demo_save_dir> 

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import open3d as o3d
import numpy as np
from termcolor import cprint
import scipy.spatial.transform as st
from pcdp.real_world.real_env_piper import RealEnv
from pcdp.real_world.teleoperation_piper import TeleoperationPiper
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
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


camera_to_base = np.array([
    [  0.007131,  -0.91491,    0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
    [  0.,         0. ,        0. ,        1.      ]
    ])

workspace_bounds = np.array([
    [-0.000, 0.740],    # X range (milli meters)
    [-0.400, 0.350],    # Y range (milli meters)
    [-0.100, 0.600]     # Z range (milli meters)
])

d405_to_eef = np.array([
    [  -0.000,   -0.839,   0.545,   -0.0650],
    [   1.000,   -0.000,  -0.000,  0.015],
    [   0.000,   0.545,   0.839,  0.035],
    [   0.000,   0.000,   0.000,   1.000]
])

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])


@click.command()
@click.option('--output', '-o', required=True, default ="demo_dataset", help="Directory to save demonstration dataset.")
@click.option('--visual', default=False, type=bool, help="Which visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--save-data', is_flag=True, default=True, help="Enable saving episode data (pointcloud, robot state).")
def main(output, visual, init_joints, frequency, command_latency, save_data):
    dt = 1/frequency

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            TeleoperationPiper(shm_manager=shm_manager) as ms, \
            RealEnv(
                output_dir=output,
                # recording resolution
                frequency=frequency,
                init_joints=init_joints,
                orbbec_mode="C2D",
                shm_manager=shm_manager,
                save_data=save_data
            ) as env:
            cv2.setNumThreads(1)
            
            include_d405 = False

            base_pose = [0.054952, 0.0, 0.493991, 0.0, np.deg2rad(85.0), 0.0, 0.0]
            plan_time = mono_time.now_s() + 2.0
            env.exec_actions([base_pose], [plan_time])
            print("Moving to the base_pose, please wait...")
            
            time.sleep(2.0)
            print('Ready!')
            
            
            main_preprocessor = PointCloudPreprocessor(extrinsics_matrix=camera_to_base,
                                                workspace_bounds=workspace_bounds,
                                                enable_filter=True,
                                                enable_sampling=False,
                                                )
            
            state = env.get_robot_state()
            target_pose = base_pose.copy()
            t_start = mono_time.now_s()
            iter_idx = 0
            stop = False
            is_recording = False
            visual_= visual
            
            stage=0

            if visual_:
                # open3d visualization
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name='default', width=1280, height=720)
                opt = vis.get_render_option()
                opt.point_size = 2.0
                opt.background_color = np.array([1., 1.0, 1.0])
                pcd = o3d.geometry.PointCloud()
                # Add a coordinate frame
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                vis.add_geometry(coordinate_frame)
            first = True

            duration = np.array([])
            t0=0

            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()
                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(mono_time.now_s()+2*dt)
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                    elif isinstance(key_stroke, KeyCode) and key_stroke.char is not None and key_stroke.char.isdigit():
                        # 문자 '0'~'9' -> 정수 0~9
                        stage = int(key_stroke.char)
                        print(f"[Stage] set to {stage}")
                
                # visualize
                # if is_recording:
                    # print("recoding!")
                episode_id = env.recorder.n_episodes
                panel = make_status_panel(
                    episode_id=episode_id,
                    recording=is_recording,
                    stage=stage,
                    extra_text="C: Start  S: stop  Q: quit 0-9: stage"
                )
                cv2.imshow("Status", panel)
                cv2.waitKey(1)

                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                
                
                if visual_:
                    # =========== Point Cloud Processing ===========
                    # 1. Process main (Orbbec) point cloud
                    main_pc = obs["main_pointcloud"][-1].copy()
                    main_pc_processed = main_preprocessor.process(main_pc)

                    # 2. Process eef (D405) point cloud
                    if include_d405:
                        d405_pc = obs['eef_pointcloud'][-1].copy()
                    robot_eef_pose = obs['robot_eef_pose'][-1] # 6D pose [x,y,z,rx,ry,rz]

                    # Convert 6D eef pose to 4x4 matrix
                    T_base_eef = np.eye(4)
                    T_base_eef[:3, :3] = st.Rotation.from_euler('xyz', robot_eef_pose[3:]).as_matrix()
                    T_base_eef[:3, 3] = robot_eef_pose[:3]

                    # Assemble full TF for D405
                    T_platform_d405 = robot_to_base @ T_base_eef @ d405_to_eef

                    # Manually apply transformation to D405 points
                    if include_d405:
                        d405_pc_xyz = d405_pc[:,:3]
                        d405_pc_rgb = d405_pc[:,3:6]
                        d405_pc_h = np.hstack((d405_pc_xyz, np.ones((d405_pc_xyz.shape[0], 1))))
                        d405_pc_transformed_h = (T_platform_d405 @ d405_pc_h.T).T
                        d405_pc_transformed = np.hstack((d405_pc_transformed_h[:,:3], d405_pc_rgb))

                        # Manually crop D405 points
                        mask = (
                            (d405_pc_transformed[:, 0] >= workspace_bounds[0][0]) & 
                            (d405_pc_transformed[:, 0] <= workspace_bounds[0][1]) &
                            (d405_pc_transformed[:, 1] >= workspace_bounds[1][0]) & 
                            (d405_pc_transformed[:, 1] <= workspace_bounds[1][1]) &
                            (d405_pc_transformed[:, 2] >= workspace_bounds[2][0]) & 
                            (d405_pc_transformed[:, 2] <= workspace_bounds[2][1])
                        )
                        d405_pc_processed = d405_pc_transformed[mask]

                    # Orbbec PCD
                    pcd_orbbec = o3d.geometry.PointCloud()
                    pcd_orbbec.points = o3d.utility.Vector3dVector(main_pc_processed[:,:3])
                    pcd_orbbec.colors = o3d.utility.Vector3dVector(main_pc_processed[:,3:6])

                    if include_d405:
                        # D405 PCD (color it blue)
                        pcd_d405 = o3d.geometry.PointCloud()
                        pcd_d405.points = o3d.utility.Vector3dVector(d405_pc_processed[:,:3])
                        pcd_d405.colors = o3d.utility.Vector3dVector(d405_pc_processed[:,3:6])

                    # Combine and update
                    pcd.points = pcd_orbbec.points
                    pcd.colors = pcd_orbbec.colors
                    if include_d405:
                        pcd.points.extend(pcd_d405.points)
                        pcd.colors.extend(pcd_d405.colors)

                    if first:
                        vis.add_geometry(pcd)
                        ctr = vis.get_view_control()
                        bbox = pcd.get_axis_aligned_bounding_box()
                        ctr.set_lookat(bbox.get_center())
                        ctr.set_front([0.0, 0.0, -1.0])
                        ctr.set_up([0.0, -1.0, 0.0])
                        ctr.set_zoom(0.4)
                        first = False
                    else:
                        vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                
                diff = mono_time.now_ms() - t0
                if diff > 120:
                    duration = np.append(duration, diff)
                t0 = mono_time.now_ms()
                
                precise_wait(t_sample)
                # get teleop command
                target_pose = ms.get_motion_state()
                action_to_record = obs['robot_eef_pose'][-1]
                # execute teleop command
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
            
            if duration.size > 1:
                duration = duration[1:]
                print(f"""[demo_real_piper]
                        cnt: {len(duration)} ea
                        mean: {duration.mean():.4f} ms
                        max: {duration.max():.4f} ms
                        min: {duration.min():.4f} ms""")

# %%
if __name__ == '__main__':
    main()
