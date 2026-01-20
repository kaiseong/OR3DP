# recorder.py
from typing import Optional, Callable, Dict
import pathlib
import os
import time
import threading
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import zarr
import numcodecs
from collections import deque
import shutil
import queue
from pcdp.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from pcdp.real_world.single_orbbec import SingleOrbbec
from pcdp.real_world.single_realsense import SingleRealSense
from pcdp.real_world.piper_interpolation_controller import PiperInterpolationController
from pcdp.real_world.video_recorder import VideoRecorder
from pcdp.common.replay_buffer import ReplayBuffer
import pcdp.common.mono_time as mono_time
from termcolor import cprint

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

class RecorderCommand:
    START = 0
    STOP = 1
    DROP = 2



class EpisodeStreamer:
    def __init__(
            self,
            episode_dir: pathlib.Path,
            save_batch_size: int = 30,
            compression_level: int =3,
            save_data: bool = False,
    ):
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.save_batch_size = save_batch_size
        self.save_data = save_data
        self.compression_level = compression_level

        # Batch buffers
        self.obs_batch_buffer = []
        self.action_batch_buffer = []
        
        # Zarr arrays
        self.obs_arrays = {}
        self.action_arrays = {}
        self.obs_total_frames = 0
        self.action_total_frames = 0

        # Replay buffers
        obs_zarr_path = str(self.episode_dir.joinpath('obs_replay_buffer.zarr').absolute())
        action_zarr_path = str(self.episode_dir.joinpath('action_replay_buffer.zarr').absolute())
        self.obs_replay_buffer = ReplayBuffer.create_from_path(zarr_path=obs_zarr_path, mode='a')
        self.action_replay_buffer = ReplayBuffer.create_from_path(zarr_path=action_zarr_path, mode='a')
        cprint(f"[EpisodeStreamer] Created for {episode_dir}", "blue", attrs=["bold"])
    
    def add_obs_data(self, obs_data: dict):
        self.obs_batch_buffer.append(obs_data)
        if len(self.obs_batch_buffer) >= self.save_batch_size:
            self._flush_obs_batch()
    
    def add_action_data(self, action_data: dict):
        self.action_batch_buffer.append(action_data)
        if len(self.action_batch_buffer) >= self.save_batch_size:
            self._flush_action_batch()
    
    def _flush_obs_batch(self):
        if not self.obs_batch_buffer:
            return
        
        try:
            batch_data = {}
            keys_to_save = self.obs_batch_buffer[0].keys()
            for key in keys_to_save:
                batch_data[key] = np.array([obs[key] for obs in self.obs_batch_buffer])

            self._append_to_zarr_arrays(batch_data, 'obs')
            self.obs_total_frames += len(self.obs_batch_buffer)
            
            self.obs_batch_buffer.clear()
        except Exception as e:
            cprint(f"[Recorder] Error flushing obs batch: {e}", "cyan", attrs=["bold"])

    def _flush_action_batch(self):
        if not self.action_batch_buffer:
            return
        
        try:
            batch_data = {}
            for key in ['action', 'timestamp', 'stage']:
                batch_data[key] = np.array([action[key] for action in self.action_batch_buffer])
            
            self._append_to_zarr_arrays(batch_data, 'action')
            self.action_total_frames += len(self.action_batch_buffer)
            
            self.action_batch_buffer.clear()
        except Exception as e:
            cprint(f"[Recorder] Error flushing action batch: {e}", "cyan", attrs=["bold"])

    def _append_to_zarr_arrays(self, batch_data: dict, data_type: str):
        if data_type == 'obs':
            arrays = self.obs_arrays
            replay_buffer = self.obs_replay_buffer
        elif data_type == 'action':
            arrays = self.action_arrays
            replay_buffer = self.action_replay_buffer
        
        
        batch_size = len(next(iter(batch_data.values())))

        for key, value in batch_data.items():
            if key not in arrays:
                initial_shape = (batch_size,) + value.shape[1:]
                chunks = (min(256, batch_size),) + value.shape[1:]

                zarr_group = replay_buffer.data

                compressor = numcodecs.Blosc(
                    cname='zstd', 
                    clevel=self.compression_level, 
                    shuffle=numcodecs.Blosc.BITSHUFFLE
                )
                
                
                arrays[key] = zarr_group.zeros(
                    name=key,
                    shape=initial_shape,
                    chunks=chunks,
                    dtype=value.dtype,
                    compressor=compressor
                )
            else:
                arr = arrays[key]
                new_shape = (arr.shape[0] + batch_size,) + arr.shape[1:]
                arr.resize(new_shape)
        
            arr = arrays[key]
            arr[-batch_size:] = value.astype(arr.dtype, copy=False)

    def finalize_episode(self):
        if self.obs_batch_buffer:
            self._flush_obs_batch()
        if self.action_batch_buffer:
            self._flush_action_batch()
        
        if self.obs_total_frames > 0:
            episode_ends = self.obs_replay_buffer.episode_ends
            if len(episode_ends) == 0:
                episode_ends.resize(1)
                episode_ends[0] = self.obs_total_frames
            else:
                episode_ends.resize(len(episode_ends)+1)
                episode_ends[-1] = episode_ends[-2] + self.obs_total_frames
        
        if self.action_total_frames > 0:
            episode_ends = self.action_replay_buffer.episode_ends
            if len(episode_ends) == 0:
                episode_ends.resize(1)
                episode_ends[0] = self.action_total_frames
            else:
                episode_ends.resize(len(episode_ends)+1)
                episode_ends[-1] = episode_ends[-2] + self.action_total_frames
        cprint(f"[Recorder] Episode finalized with {self.obs_total_frames} obs frames and {self.action_total_frames} action frames", 
                "cyan", attrs=["bold"])

class Recorder(mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        orbbec: SingleOrbbec,
        robot: PiperInterpolationController,
        output_dir: str,
        d405: SingleRealSense = None,
        compression_level: int = 3,
        frequency: float = 200.0,  
        max_buffer_size: int = 30,
        save_batch_size: int = 1,
        record_rgb: bool = True,
        rgb_fps: int = 30,
        save_data: bool = False
    ):
        super().__init__(name="Recorder")
        
        self.shm_manager = shm_manager
        self.orbbec = orbbec
        self.d405 = d405
        self.robot = robot
        self.output_dir = pathlib.Path(output_dir)
        self.compression_level = compression_level
        self.polling_dt = 1.0 / frequency  
        self.max_buffer_size = max_buffer_size
        self.save_batch_size = save_batch_size
        self.save_data = save_data
        self.writer_queue = queue.Queue(maxsize=1024)
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True
        )
        
        self.episode_counter = self._get_latest_episode_id()
        self.episode_streamer = None

        self.recording = False
        self.start_time = None
        self.episode_id = None
        
        self.robot_buffer = deque(maxlen=max_buffer_size)
        self.action_buffer = deque(maxlen=max_buffer_size)
        self.orbbec_buffer = deque(maxlen=max_buffer_size)
        self.d405_buffer = None
        if self.d405 is not None:
            self.d405_buffer = deque(maxlen=max_buffer_size * 2) 

        self.last_orbbec_timestamp = -1
        self.last_robot_timestamp = -1
        self.last_d405_timestamp = -1
        
        self.stop_event = mp.Event()
        self.recording_event = mp.Event()
        
        command_examples = {
            'cmd': RecorderCommand.START,  # 'start' or 'stop' 'drop'
            'start_time': 0.0,
            'episode_id': 0
        }
        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=command_examples,
            buffer_size=64
        )

        action_example = {
            'action': np.zeros(7, dtype = np.float64),
            'timestamp': 0.0,
            'stage': 0
        }

        self.action_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=action_example,
            buffer_size=256
        )

        self.orbbec_video_recorder: Optional[VideoRecorder] = None
        self.d405_video_recorder: Optional[VideoRecorder] = None

        if record_rgb:
            if self.orbbec is not None and self.orbbec.rgb_ring_buffer is not None:
                self.orbbec_video_recorder = VideoRecorder(
                    shm_manager=shm_manager,
                    rgb_queue=self.orbbec.rgb_ring_buffer,
                    fps=rgb_fps,
                )
            if self.d405 is not None and self.d405.rgb_ring_buffer is not None:
                self.d405_video_recorder = VideoRecorder(
                    shm_manager=shm_manager,
                    rgb_queue=self.d405.rgb_ring_buffer,
                    fps=self.d405.put_fps
                )
    
    def _get_latest_episode_id(self):
        max_id = -1

        # Check Zarr data directories
        zarr_dir = self.output_dir
        if zarr_dir.exists():
            for item in zarr_dir.iterdir():
                if item.is_dir() and item.name.startswith('episode_'):
                    try:
                        episode_id = int(item.name.split('_')[1])
                        max_id = max(max_id, episode_id)
                    except (ValueError, IndexError):
                        continue
        
        # Check video directories
        video_dir = self.output_dir.parent / 'videos'
        if video_dir.exists():
            for item in video_dir.iterdir():
                if item.is_dir():
                    try:
                        # Videos are in folders named '0000', '0001', etc.
                        episode_id = int(item.name)
                        max_id = max(max_id, episode_id)
                    except (ValueError, IndexError):
                        continue

        return max_id + 1

    def get_episode_dir(self, episode_id: int):
        return self.output_dir / f"episode_{episode_id:04d}"
    

    def start_recording(self, start_time: float, episode_id: int):
        try:
            self.command_queue.put({
                'cmd': RecorderCommand.START,
                'start_time': start_time,
                'episode_id': episode_id
            })
        except Full:
            cprint("[Recorder] Warning: Command queue full", "cyan", attrs=["bold"])
    
    def stop_recording(self):
        try:
            self.command_queue.put({
                'cmd': RecorderCommand.STOP,
                'start_time': 0.0,
                'episode_id': 0
            })
        except Full:
            cprint("[Recorder] Warning: Command queue full", "cyan", attrs=["bold"])
    
    def drop_episode(self):
        try:
            self.command_queue.put({
                'cmd': RecorderCommand.DROP,
                'start_time': 0.0,
                'episode_id': 0
            })
        except Full:
            cprint("[Recorder] Warning: Command queue full", "cyan", attrs=["bold"])

    def add_action(self, action: np.ndarray, timestamp: float, stage: int =0):
        try:
            self.action_queue.put({
                'action': action,
                'timestamp': timestamp,
                'stage': stage
            })
        except Full:
            cprint(f"[Recorder] Action queue full", "cyan", attrs=["bold"])

    
    def start(self, wait=True):
        super().start()
        if self.orbbec_video_recorder:
            self.orbbec_video_recorder.start()
        if self.d405_video_recorder:
            self.d405_video_recorder.start()
        if wait:
            self.start_wait()
    
    def start_wait(self):
        time.sleep(0.1)
    
    def stop(self, wait=True):
        self.stop_process()
        if wait:
            self.join()


    def stop_process(self):
        self.stop_event.set()
        if self.orbbec_video_recorder:
            self.orbbec_video_recorder.stop()
        if self.d405_video_recorder:
            self.d405_video_recorder.stop()

    def _read_orbbec_data(self) -> bool: 
        try:
            all_data=self.orbbec.get(k=1)
            
            if all_data is not None and 'timestamp' in all_data:
                timestamps = all_data['timestamp']
                new_data_added = False
                for i, timestamp in enumerate(timestamps):
                    if timestamp > self.last_orbbec_timestamp:
                        orbbec_data = {
                            'pointcloud': all_data['pointcloud'][i],
                            'timestamp': timestamp,
                            'camera_capture_timestamp': all_data['camera_capture_timestamp'][i]
                        }
                        self.orbbec_buffer.append(orbbec_data)
                        self.last_orbbec_timestamp = timestamp
                        new_data_added = True
                return new_data_added
        except Exception as e:
            cprint(f"[Recorder] Error reading orbbec data: {e}", "cyan", attrs=["bold"])
        return False
    
    def _read_d405_data(self) -> bool:
        if self.d405 is None:
            return False
        try:
            all_data = self.d405.get(k=4)
            if all_data is not None and 'timestamp' in all_data:
                timestamps = all_data['timestamp']
                new_data_added = False
                for i, timestamp in enumerate(timestamps):
                    if timestamp > self.last_d405_timestamp:
                        d405_data = {
                            'pointcloud': all_data['pointcloud'][i],
                            'timestamp': timestamp,
                            'camera_capture_timestamp': all_data['camera_capture_timestamp'][i]
                        }
                        self.d405_buffer.append(d405_data)
                        self.last_d405_timestamp = timestamp
                        new_data_added = True
                return new_data_added
        except Exception as e:
            cprint(f"[Recorder] Error reading d405 data: {e}", "cyan", attrs=["bold"])
        return False


    def _read_robot_data(self) -> bool:
        try:
            # Get latest data from robot
            all_robot_data = self.robot.get_all_state()
            if all_robot_data is not None and 'robot_receive_timestamp' in all_robot_data:
                timestamps = all_robot_data['robot_receive_timestamp']
                
                new_data_added = False
                for i, timestamp in enumerate(timestamps):
                    if timestamp > self.last_robot_timestamp:
                        robot_data = {
                            'robot_eef_pose': all_robot_data.get('ArmEndPoseMsgs', [np.zeros(6)])[i],
                            'robot_joint': all_robot_data.get('ArmJointMsgs', [np.zeros(6)])[i],
                            'robot_gripper': all_robot_data.get('ArmGripperMsgs', [np.zeros(3)])[i],
                            'robot_eef_target': all_robot_data.get('TargetEndPose', [np.zeros(6)])[i],
                            'robot_receive_timestamp': timestamp
                        }
                        self.robot_buffer.append(robot_data)
                        new_data_added = True
                        self.last_robot_timestamp = timestamp
                return new_data_added
        except Exception as e:
            cprint(f"[Recorder] Error reading robot data: {e}", "cyan")
        return False
    
    def _find_previous_robot_data(self, orbbec_timestamps: np.ndarray) -> np.ndarray:
        if len(self.robot_buffer) == 0:
            return np.array([])

        robot_timestamps = np.array([r['robot_receive_timestamp'] for r in self.robot_buffer])
        indices = np.searchsorted(robot_timestamps, orbbec_timestamps, side='right') - 1

        return indices

    def _find_previous_d405_data(self, orbbec_timestamps: np.ndarray) -> np.ndarray:
        if len(self.d405_buffer) == 0:
            return np.array([])

        d405_timestamps = np.array([d['timestamp'] for d in self.d405_buffer])
        indices = np.searchsorted(d405_timestamps, orbbec_timestamps, side='right') - 1
        return indices

    def _process_new_data(self):
        processed = 0
        MAX_BATCH = 3

        
        orbbec_data_list = []
        while processed < MAX_BATCH:
            try:
                orbbec_data = self.orbbec_buffer.popleft()
                if orbbec_data['timestamp'] >= self.start_time:
                    orbbec_data_list.append(orbbec_data)
                    processed += 1
                elif not self.recording:
                    processed += 1
                else:
                    processed += 1
            except IndexError:
                break
        
        if not orbbec_data_list:
            return
        
        orbbec_timestamps = np.array([od['timestamp'] for od in orbbec_data_list])
        robot_indices = self._find_previous_robot_data(orbbec_timestamps)

        d405_indices = None
        if self.d405 is not None:
            d405_indices = self._find_previous_d405_data(orbbec_timestamps)

        for i, orbbec_data in enumerate(orbbec_data_list):
            robot_idx = robot_indices[i]
            if robot_idx < 0:
                cprint(f"CRITICAL: Data sync failed for orbbec_timestamp {orbbec_data['timestamp']}. Robot data is missing.", "red", attrs=["bold"])
                continue
            
            robot_data = self.robot_buffer[robot_idx]

            obs_data={
                'pointcloud': orbbec_data['pointcloud'],
                'robot_eef_pose': robot_data['robot_eef_pose'],
                'robot_joint': robot_data['robot_joint'],
                'robot_gripper': robot_data['robot_gripper'],
                'robot_eef_target': robot_data['robot_eef_target'],
                'align_timestamp': orbbec_data['timestamp'],
                'robot_timestamp': robot_data['robot_receive_timestamp'],
                'orbbec_capture_timestamp': orbbec_data['camera_capture_timestamp']
            }

            if self.d405 is not None:
                d405_idx = d405_indices[i]
                if d405_idx < 0:
                    cprint(f"CRITICAL: Data sync failed for orbbec_timestamp {orbbec_data['timestamp']}. D405 data is missing.", "red", attrs=["bold"])
                    continue
                d405_data = self.d405_buffer[d405_idx]
                obs_data['eef_pointcloud'] = d405_data['pointcloud']
                obs_data['d405_timestamp'] = d405_data['timestamp']
                obs_data['d405_capture_timestamp'] = d405_data['camera_capture_timestamp']
            
            if self.episode_streamer is not None:
                self.writer_queue.put_nowait({'type': 'obs', 'data': obs_data})

    def _process_action_queue(self):
        try:
            actions = self.action_queue.get_all()
            if len(actions['action']) > 0 and self.episode_streamer:
                for i in range(len(actions['action'])):
                    action_data = {
                        'action': actions['action'][i],
                        'timestamp': actions['timestamp'][i],
                        'stage': actions['stage'][i]
                    }
                    self.writer_queue.put_nowait({'type':'action', 'data': action_data})
        except Empty:
            pass
        except Exception as e:
            cprint(f"[Recorder] Error processing actions: {e}", "cyan", attrs=["bold"])

    
    def _process_commands(self):
        try:
            commands = self.command_queue.get_all()
            n_cmd = len(commands['cmd'])
            
            for i in range(n_cmd):
                cmd = commands['cmd'][i]
                
                if cmd == RecorderCommand.START:
                    if not self.recording:
                        
                        self.start_time = commands['start_time'][i]
                        self.episode_id = commands['episode_id'][i]
                        self.episode_counter = self.episode_id + 1

                        if self.save_data:
                            episode_dir = self.get_episode_dir(self.episode_id)
                            self.episode_streamer = EpisodeStreamer(
                                episode_dir, 
                                self.save_batch_size, 
                                compression_level=self.compression_level
                            )
                        else:
                            self.episode_streamer = None

                        self.recording = True
                        self.recording_event.set()

                        video_dir = self.output_dir.parent / 'videos'
                        video_episode_dir = video_dir / f'{self.episode_id:04d}'
                        video_episode_dir.mkdir(parents=True, exist_ok=True)

                        if self.orbbec_video_recorder:
                            video_path = video_episode_dir / 'orbbec_video.mp4'
                            self.orbbec_video_recorder.start_episode_recording(str(video_path), self.start_time)
                        
                        if self.d405_video_recorder:
                            video_path = video_episode_dir / 'd405_video.mp4'
                            self.d405_video_recorder.start_episode_recording(str(video_path), self.start_time)

                        self.orbbec_buffer.clear()
                        if self.d405 is not None:
                            self.d405_buffer.clear()
                        self.robot_buffer.clear()

                        cprint(f"[Recorder] Started recording episode {self.episode_id} from time {self.start_time}", 
                                "cyan", attrs=["bold"])
                
                elif cmd == RecorderCommand.STOP:
                    if self.recording:
                        if self.episode_streamer:
                            self._process_new_data()
                            self.writer_queue.put({'type': "FLUSH"})
                            self.writer_queue.join()
                            self.episode_streamer.finalize_episode()
                            self.episode_streamer = None

                        # Always stop recording and video capture
                        self.recording = False
                        self.recording_event.clear()
                        if self.orbbec_video_recorder:
                            self.orbbec_video_recorder.stop_episode_recording()
                        if self.d405_video_recorder:
                            self.d405_video_recorder.stop_episode_recording()
                        cprint(f"[Recorder] saved episode {self.episode_id}.", "cyan", attrs=["bold"])
                
                elif cmd == RecorderCommand.DROP:
                    if self.recording and self.episode_streamer:
                        self._process_new_data()
                        self.writer_queue.put({'type': "FLUSH"})
                        self.writer_queue.join()
                        self.episode_streamer = None
                        self.recording = False
                        self.recording_event.clear()
                        if self.orbbec_video_recorder:
                            self.orbbec_video_recorder.stop_episode_recording()
                        if self.d405_video_recorder:
                            self.d405_video_recorder.stop_episode_recording()
                    

                    if self.episode_counter > 0:
                        drop_episode_id = self.episode_counter -1
                        if self.episode_id is not None and self.episode_id == drop_episode_id:
                            self.episode_id = None
                        episode_dir = self.get_episode_dir(drop_episode_id)
                        video_dir = self.output_dir.parent / 'videos' / f'{drop_episode_id:04d}'
                        
                        if episode_dir.exists():
                            shutil.rmtree(episode_dir)
                            cprint(f"[Recorder] Dropped episode data: {episode_dir}", "cyan", attrs=["bold"])
                        if video_dir.exists():
                            shutil.rmtree(video_dir)
                            cprint(f"[Recorder] Dropped episode video: {video_dir}", "cyan", attrs=["bold"])
                        self.episode_counter = drop_episode_id
                        
        except Empty:
            pass
        except Exception as e:
            cprint(f"[Recorder] Error processing commands: {e}", "cyan")
            import traceback
            traceback.print_exc()
    
    def _writer_loop(self):
        buffer = []
        while True:
            item = self.writer_queue.get()
            if item['type'] == 'FLUSH':
                if buffer:
                    self._save_batch(buffer)
                    buffer.clear()
                self.writer_queue.task_done()
                continue

            if item['type'] == 'TERMINATE':
                if buffer:
                    self._save_batch(buffer)
                self.writer_queue.task_done()
                break

            buffer.append(item)     

            if len(buffer) >= self.save_batch_size:
                self._save_batch(buffer)
                buffer.clear() 
            self.writer_queue.task_done()
        
        

    def _save_batch(self, batch):
        for item in batch:
            if not self.episode_streamer:
                continue
            if item['type'] == 'obs':
                self.episode_streamer.add_obs_data(item["data"])
            elif item['type'] == 'action':
                self.episode_streamer.add_action_data(item["data"])
            

    def run(self):
        cprint(f"[Recorder] Starting recorder process", "cyan", attrs=["bold"])
        cprint(f"[Recorder] now episode: {self.n_episodes}", "cyan", attrs=["bold"])
        self.writer_thread.start()
        try:
            while not self.stop_event.is_set():
                
                start_time = mono_time.now_s()
                
                self._process_commands()
                
                orbbec_updated = self._read_orbbec_data()
                d405_updated = self._read_d405_data()
                robot_updated = self._read_robot_data()
                self._process_action_queue()
                
                if self.recording and orbbec_updated:
                    self._process_new_data()
                    
                elapsed = mono_time.now_s() - start_time
                sleep_time = max(0, self.polling_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            cprint(f"[Recorder] Error in main loop: {e}", "cyan", attrs=["bold"])
            import traceback
            traceback.print_exc()
        finally:
            if self.recording and self.episode_streamer:
                self._process_new_data()
                self.writer_queue.put({'type':'FLUSH'})
                self.writer_queue.join()
                self.episode_streamer.finalize_episode()
            self.writer_queue.put({'type':'TERMINATE'})
            self.writer_queue.join()
            self.writer_thread.join(timeout=1.0)
            cprint("[Recorder] Recorder process ended", "cyan", attrs=["bold"])
    
    @property
    def is_recording(self):
        return self.recording_event.is_set()
    
    @property
    def n_episodes(self):
        return self._get_latest_episode_id()
    

    def get_episode_list(self):
        episodes = []
        if self.output_dir.exists():
            for item in sorted(self.output_dir.iterdir()):
                if item.is_dir() and item.name.startswith('episode'):
                    ep_id_str = item.name.split('episode_')[-1]
                    try:
                        episodes.append(int(ep_id_str))
                    except ValueError:
                        continue
        episodes.sort()
        return episodes
