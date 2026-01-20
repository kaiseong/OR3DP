# teleoperation_piper.py
import multiprocessing as mp
import numpy as np
import time
import pcdp.common.mono_time as mono_time
from multiprocessing.managers import SharedMemoryManager
from pcdp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from piper_sdk import *
from termcolor import cprint

class TeleoperationPiper(mp.Process):
    def __init__(self,
                shm_manager: SharedMemoryManager,
                frequency: float=200,
                get_max_k: int = 30,
                dtype=np.float64,
                threshold: float=10.0,
                ):
        super().__init__(name="TeleoperationPiper")

        self.frequency = frequency
        self.dtype = dtype
        self.threshold = threshold
        action_example = {
            'action': np.zeros(7, dtype=self.dtype),
            'timestamp': mono_time.now_s()
        }

        self.action_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=action_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

    def get_motion_state(self):
        state = self.action_ring_buffer.get()
        # [Tx, Ty, Tz, Rx, Ry, Rz, Grip]
        action = np.array(state['action'][:7], dtype=self.dtype)
        return action
    
    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
        
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def run(self):
        try:
            piper = C_PiperInterface_V2(can_name = "can_master")
            piper.ConnectPort()
            piper.EnableArm(7)
            self.ready_event.set()
        except Exception as e:
            cprint(f"[Teleoper Piper] Failed to connect to Piper: {e}", "red", attrs=["bold"])
            self.ready_event.set()
            return

        # main loop
        try:
            dt = 1.0 / self.frequency

            while not self.stop_event.is_set():
                t_start = mono_time.now_s()

                raw_state = piper.GetArmEndPoseMsgs()
                raw_gripper =  piper.GetArmGripperMsgs()
                state=np.asarray(raw_state, dtype=self.dtype)
                gripper = -2.4 + (101.4/72.75) * (raw_gripper[0]+1.75)
                state[:3] = state[:3] * 1e-3
                state[3:] = np.deg2rad(state[3:])
                state=np.append(state, gripper)
                
                
                self.action_ring_buffer.put({
                    'action': state,
                    'timestamp': t_start
                })

                elapsed = mono_time.now_s() - t_start
                sleep_time = max(0, dt-elapsed)
                if sleep_time>0:
                    time.sleep(sleep_time)
        finally:
            # piper.DisableArm(7)
            time.sleep(1)
            piper.DisconnectPort()
            self.ready_event.set()




