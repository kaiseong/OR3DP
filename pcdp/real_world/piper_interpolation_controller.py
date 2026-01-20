# PiperInterpolationController.py
import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from typing import (Optional, List)
from piper_sdk import *
from pcdp.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from pcdp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from pcdp.common.no_pose_trajectory_interpolator import PoseTrajectoryInterpolator
# from pcdp.common.only_pose_trajectory_interpolator import PoseTrajectoryInterpolator
# from pcdp.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import pcdp.common.mono_time as mono_time
from termcolor import cprint




class Command(enum.Enum):
    """Command enum for the interpolation controller."""
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class PiperInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    
    def __init__(self,
            shm_manager: SharedMemoryManager,
            # Controller parameters
            frequency = 200,
            lookahead_time=0.1,
            gain = 300,
            max_pos_speed=30,
            max_rot_speed=170,
            launch_timeout=3,
            payload_mass=None,
            joints_init=None,
            soft_real_time=False,
            debug=False,
            receive_keys=None,
            get_max_k=128,
            mode="EndPose",
            ):
        """
        frequency: piper 200 Hz
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: rad/s
        max_rot_speed: rad/s
        payload_mass: float
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 0 < max_pos_speed and 0 < max_rot_speed

        if payload_mass is not None:
            assert 0 <= payload_mass <= 1.5
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)
        else:
            joints_init = np.array(np.zeros(6,))
            

        super().__init__(name="PiperInterpolationController")
        
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.payload_mass = payload_mass
        self.joints_init = joints_init
        self.soft_real_time = soft_real_time
        self.verbose = debug
        self.mode = mode
        
        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((7,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ArmEndPoseMsgs',
                'ArmJointMsgs',
                'ArmGripperMsgs',
                'TargetEndPose',
            ]
        shape_map = {
            'ArmEndPoseMsgs': (6,),   # X Y Z RX RY RZ
            'ArmJointMsgs':   (6,),   # j1~j6
            'ArmGripperMsgs': (3,),   # angle effort status
            'TargetEndPose': (7,)
        }
        example = {k: np.zeros(shape_map[k], dtype=np.float64) for k in receive_keys}
        example['robot_receive_timestamp'] = mono_time.now_s()
        

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
    
    # =========== launch method ============ 
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PiperPositionalController] Controller process spawned at {self.pid}")
        
    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()
    
    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # =========== unit convert method ============
    def _sdk_pose_to_vec(self, raw):
        # raw is [mm] and [deg]
        v = np.asarray(raw, dtype=float)
        v[:3] *= 1e-3
        v[3:] = np.deg2rad(v[3:])
        return v

    def _rad_to_sdk_joint(self, rad: np.ndarray) -> List[int]:
        """Converts joint angles in radians to the integer format for Piper SDK."""
        return (np.rad2deg(rad) * 1e3).astype(int).tolist()

    # =========== context method ============
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, etc_val, exc_tb):
        self.stop()
    
    # =========== command method ============
    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert (duration >= (1/self.frequency))
        pose = np.asarray(pose, dtype=float)
        assert pose.shape == (7,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': float(duration),
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > mono_time.now_s()-0.002 # 2ms tolerance
        pose = np.asarray(pose,dtype=float)
        assert pose.shape == (7,)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': float(target_time)
        }   
        self.input_queue.put(message)

    # =========== receive APIs ============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # =========== main loop in process ============
    def run(self):
        durations = np.array([], dtype=np.float32)
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
        
        # start
        try:
            piper = C_PiperInterface_V2("can_slave")
            piper.ConnectPort()
            piper.SetSDKJointLimitParam('j4', -1.7977, 1.7977)
            piper.SetSDKJointLimitParam('j5', -1.4265, 1.4265)
            piper.SetSDKJointLimitParam('j6', -2.0071, 2.2166)
        except Exception as e:
            cprint(f"[Piper_Controller] Failed to connect to Piper: {e}", "magenta", attrs=["bold"])
            self.ready_event.set()
            return
        piper.EnableArm(7)

        enable_flag = False
        timeout = 5

        start_time = mono_time.now_s()

        elapsed_time_flag = False
        while not (enable_flag):
            elapsed_time = mono_time.now_s() - start_time
            print("--------------------")
            enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("상태 활성화:",enable_flag)
            piper.EnableArm(7)
            piper.GripperCtrl(0,5000,0x01, 0)
            print("--------------------")
            if elapsed_time > timeout:
                print("시간초과....")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)
            pass
        if(elapsed_time_flag):
            print("프로그램 timeout으로 종료합니다")
            exit(0)

        try:
            if self.verbose:
                cprint(f"[Piper_Controller] Connect to robot: {piper.GetCanFps()} Hz",
                        "magenta", attrs=["bold"])

            #set parameters
            if self.payload_mass is not None:
                if self.payload_mass <= 0.75:
                    piper.ArmParamEnquiryAndConfig(0x00, 0x00, 0x00, 0x00, 0x01)
                elif self.payload_mass >= 1.35:
                    piper.ArmParamEnquiryAndConfig(0x00, 0x00, 0x00, 0x00, 0x02)
            
            # init pose
            if self.joints_init is not None:
                piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                piper.JointCtrl(self._rad_to_sdk_joint(self.joints_init))
            
            # main loop
            dt = 1. / self.frequency
            
            # Read current state
            curr_joints = np.deg2rad(np.asarray(piper.GetArmJointMsgs()))
            curr_pose_6d = self._sdk_pose_to_vec(piper.GetArmEndPoseMsgs())
            # gripper state is not available from GetArmEndPoseMsgs, so append a default value (1.0 for open)
            curr_pose = np.append(curr_pose_6d, 1.0)
            
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            
            # Initialize interpolator with current EEF pose
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            
            
            # Initialize last known joint angles for IK
            last_q = curr_joints

            iter_idx = 0
            keep_running=True
            cprint(f"[Piper_Controller] Main loop started.", "magenta", attrs=["bold"])
            
            debug_first = False
            
            while keep_running:
                
                debug_t_start = mono_time.now_ms()

                # start control iteration
                loop_start=time.perf_counter()

                # send command to robot
                t_now = time.monotonic()
                
                # 1. Get current robot state
                state = dict()
                for k in self.receive_keys:
                    if k == 'TargetEndPose':
                        continue
                    raw = getattr(piper, 'Get'+k)()
                    if k == 'ArmEndPoseMsgs':
                        state[k] = self._sdk_pose_to_vec(raw)
                    else:
                        state[k] = np.asarray(raw)
                
                current_joints_rad = np.deg2rad(state['ArmJointMsgs'])
                last_q = current_joints_rad
                

                # 2. Get target EEF pose from interpolator
                target_pose_vec = pose_interp(t_now)
                
                if self.mode == "EndPose":
                    pos = target_pose_vec[:3] *1e6
                    rot = np.rad2deg(target_pose_vec[3:6])*1e3
                    target= np.append(pos.astype(int), rot.astype(int))
                    target = target.tolist()
                else:
                    pos = target_pose_vec[:3]
                    rot_vec = target_pose_vec[3:6]
                    # rot = st.Rotation.from_rotvec(rot_vec).as_matrix()
                    rot = st.Rotation.from_euler('xyz', rot_vec).as_matrix()
                    target_se3 = pin.SE3(rot, pos)
                
                gripper = target_pose_vec[6]
                
                
                # 4. Calculate IK using the most recent joint state
                if self.mode == "EndPose":
                    piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                    piper.EndPoseCtrl(target)
                else:
                    target_joints = self.ik_controller.calculate_ik(target_se3, last_q)
                    
                    if target_joints is not None:
                        piper.MotionCtrl_2(0x01, 0x01, 100, 0x00) # Using a moderate speed
                        piper.JointCtrl(self._rad_to_sdk_joint(target_joints))
                    else:
                        if self.verbose:
                            cprint(f"[Piper_Controller] IK failed at t={t_now}", "red")
                
                # if gripper <= 1.1 and gripper >= 0.9:
                #     piper.GripperCtrl(0, 1000, 0x01)
                # else:
                #     piper.GripperCtrl(95000, 1000, 0x01)
                piper.GripperCtrl(int(gripper*1e3), 1000, 0x01)

                # 6. Store state in ring buffer
                state["ArmJointMsgs"] = current_joints_rad
                state["TargetEndPose"] = target_pose_vec
                state["robot_receive_timestamp"] = mono_time.now_s()
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                
                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        # stop immediately, ignore later commands
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose = target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time=t_insert
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time=target_time
                    else:
                        keep_running = False
                        break
                if iter_idx==0:
                    self.ready_event.set()
                iter_idx += 1

                if debug_first:
                    durations = np.append(durations, mono_time.now_ms()-debug_t_start)
                debug_first = True

                spent = time.perf_counter() - loop_start
                if spent < dt:
                    time.sleep(dt - spent)
                
                
        
        finally:
            print(f"Piper Controller\n\
                    duration mean: {durations.mean()}\n\
                    max: {durations.max()}\n\
                    min: {durations.min()}")
            # manditory cleanup
            piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            piper.JointCtrl([0, 0, 0, 0, 0, 0])
            time.sleep(5)
            piper.DisableArm(7)
            time.sleep(1)
            piper.DisconnectPort()
            self.ready_event.set()

            if self.verbose:
                cprint(f"[PiperPositionalController] Disconnect from robot", "magenta", attrs=["bold"])