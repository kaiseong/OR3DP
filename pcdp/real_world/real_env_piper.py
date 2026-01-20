from typing import Optional, List
import pathlib
import numpy as np
import math
from multiprocessing.managers import SharedMemoryManager
from pcdp.real_world.piper_interpolation_controller import PiperInterpolationController
from pcdp.real_world.single_orbbec import SingleOrbbec
from pcdp.real_world.single_realsense import SingleRealSense
from pcdp.real_world.recorder import Recorder
import pcdp.common.mono_time as mono_time
from termcolor import cprint
import time

DEFAULT_OBS_KEY_MAP = {
    # robot
    'ArmEndPoseMsgs': 'robot_eef_pose',
    'ArmJointMsgs': 'robot_joint',
    'ArmGripperMsgs': 'robot_gripper',
    'TargetEndPose': 'robot_eef_target',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class RealEnv:
    def __init__(self, 
            # required params
            output_dir: str,
            # env params
            frequency=10,
            n_obs_steps=2,
            # obs
            max_obs_buffer_size=30,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            # robot
            init_joints=False,
            # video capture params
            capture_fps=30,
            # vis params
            orbbec_mode = "C2D",
            # shared memory
            shm_manager=None,
            save_data: bool = False
            ):
        assert frequency <= capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()

        recorder_data_dir = output_dir.joinpath('recorder_data')
        recorder_data_dir.mkdir(parents=True, exist_ok=True)

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        
        
        
        orbbec = SingleOrbbec(
            shm_manager=shm_manager,
            rgb_resolution = (1280, 720),
            put_fps = capture_fps,
            get_max_k = max_obs_buffer_size,
            mode=orbbec_mode,
            verbose=True
        )
        
        # d405 = SingleRealSense(
        #     shm_manager=shm_manager,
        #     resolution = (424, 240),
        #     put_fps = capture_fps*2, # 60 fps
        #     get_max_k = max_obs_buffer_size*2, # 60
        #     num_downsample = 36000,
        #     verbose=False
        # )
        d405 = None

        j_init = np.array([0.0, 0.5, -1.0, 0.0, 1.0, 0.0]) 
        if not init_joints:
            j_init = None

        robot = PiperInterpolationController(
            shm_manager=shm_manager,
            # Controller parameters
            frequency=200, # Piper frequency
            lookahead_time=0.1,
            gain=300,
            launch_timeout=3,
            payload_mass=None,
            joints_init=j_init,
            soft_real_time=False,
            debug=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size
            )
        
        recorder = Recorder(
            shm_manager=shm_manager,
            orbbec=orbbec,
            d405=d405,
            robot=robot,
            output_dir=str(recorder_data_dir),
            compression_level=2,
            frequency=200.0,
            save_data=save_data
        )
        
        self.orbbec = orbbec
        self.d405 = d405
        self.robot = robot
        self.recorder = recorder
        self.capture_fps = capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        # temp memory buffers
        self.last_orbbec_data = None
        self.last_d405_data=None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready = self.orbbec.is_ready and self.robot.is_ready
        if self.d405 is not None:
            ready = ready and self.d405.is_ready
        return ready
    
    def start(self, wait=True):
        self.orbbec.start(wait=False)
        if self.d405 is not None:
            self.d405.start(wait=False)
        self.robot.start(wait=False)
        # warm up
        time.sleep(2)
        self.recorder.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        self.robot.stop(wait=False)
        self.orbbec.stop(wait=False)
        if self.d405 is not None:
            self.d405.stop(wait=False)
        self.recorder.stop_process()
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.orbbec.start_wait()
        if self.d405 is not None:
            self.d405.start_wait()
        self.robot.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.orbbec.join()
        if self.d405 is not None:
            self.d405.join()
        self.recorder.join()
        # self.realsense.stop_wait()
        
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp 
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.frequency))
        self.last_orbbec_data = self.orbbec.get(
            k=k, 
            out=self.last_orbbec_data)
        
        if self.d405 is not None:
            self.last_d405_data = self.d405.get(
                k=k*2,
                out=self.last_d405_data
            )

        
        # 200 hz, robot_receive_timestamp
        # buffer size is 30 
        last_robot_data = self.robot.get_all_state()
        
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = self.last_orbbec_data['timestamp'][-1] 
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)


        orbbec_obs = dict()
        orbbec_timestamps = self.last_orbbec_data['timestamp']
        orbbec_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(orbbec_timestamps <= t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            orbbec_idxs.append(this_idx)
        orbbec_obs['main_pointcloud'] = self.last_orbbec_data['pointcloud'][orbbec_idxs]

        if self.d405 is not None:
            d405_obs = dict()
            d405_timestamps = self.last_d405_data['timestamp']
            d405_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(d405_timestamps <= t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                d405_idxs.append(this_idx)
            d405_obs['eef_pointcloud'] = self.last_d405_data['pointcloud'][d405_idxs]


        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps <= t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        
        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]
        

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )
        

        # return obs
        obs_data = dict(orbbec_obs)
        if self.d405 is not None:
            obs_data.update(d405_obs)
        obs_data.update(robot_obs)

        
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None,
            recorded_actions: Optional[np.ndarray]=None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = mono_time.now_s()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]
        # schedule waypoints
        for i in range(len(new_actions)):
            self.robot.schedule_waypoint(
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )
            if self.recorder.is_recording:
                # action_to_record = recorded_actions[i] if recorded_actions is not None else new_actions[i]
                self.recorder.add_action(
                    action=new_actions[i],
                    timestamp=new_timestamps[i],
                    stage=new_stages[i]
                )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = mono_time.now_s()
        self.start_time = start_time

        episode_id = self.recorder.n_episodes
        self.recorder.start_recording(start_time, episode_id)
        
    
    def end_episode(self):
        "Stop recording"
        self.recorder.stop_recording()
        

    def drop_episode(self):
        # self.end_episode()
        # self.replay_buffer.drop_episode()
        self.recorder.drop_episode()
