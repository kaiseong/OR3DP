# pcdp/real_world/video_recorder.py
from typing import Optional
import multiprocessing as mp
import av
import numpy as np
import time
import queue
import pathlib
import pcdp.common.mono_time as mono_time
from multiprocessing.managers import SharedMemoryManager
from pcdp.shared_memory.shared_memory_queue import SharedMemoryQueue 
from termcolor import cprint

class VideoRecorder(mp.Process):
    def __init__(
            self, 
            shm_manager: SharedMemoryManager, 
            rgb_queue: SharedMemoryQueue,
            fps: int, 
            codec: str = 'libx264', 
            pixel_format: str = 'yuv420p'
        ):
        super().__init__(name="VideoRecorder")
        self.shm_manager = shm_manager
        self.rgb_queue = rgb_queue
        self.fps = fps
        self.codec = codec
        self.pixel_format = pixel_format

        self.stop_event = mp.Event()
        self.command_queue = mp.Queue()
        
        self.container = None
        self.stream = None
        self.is_recording = False
        
        self.frame_index = 0
        self.n_frames_recorded = 0
        self.last_processed_timestamp = -1.0
        self.last_capture_timestamp = None # 이전 프레임의 하드웨어 타임스탬프
        self.recording_start_time = None
        self.timestamp_buffer = []
        self.csv_path = None

    
    def start_episode_recording(self, video_path: str, start_time: float = None):
        pathlib.Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        self.command_queue.put(('START', (video_path, start_time)))
        cprint(f"[VideoRecorder] Start recording command sent for: {video_path}", "white", attrs=["bold"])

    def stop_episode_recording(self):
        self.command_queue.put(('STOP', None))
        cprint(f"[VideoRecorder] Stop recording command sent", "white", attrs=["bold"])
    
    def stop(self):
        self.stop_event.set()
        self.join(timeout=2.0)
        if self.is_alive():
            cprint(f"[VideoRecorder] Process did not termminate", "red", attrs=["bold"])
            self.terminate()
        cprint(f"[VideoRecorder] Process stopped", "white", attrs=["bold"])


    def run(self):
        thread_name = self.name
        cprint(f"[{thread_name}] VideoRecorder process started", "white", attrs=['bold'])
        try:
            while not self.stop_event.is_set():
                start_time = mono_time.now_s()
                try:
                    cmd, arg = self.command_queue.get_nowait()
                    if cmd == 'START':
                        video_path, start_time = arg
                        self._start_recording_internal(video_path, start_time)
                    elif cmd == 'STOP':
                        self._stop_recording_internal()
                except queue.Empty:
                    pass
                
                if self.is_recording:
                    try:
                        frame_data = self.rgb_queue.get()
                        current_timestamp = frame_data.get('timestamp', -1.0)
                        if current_timestamp >= self.recording_start_time:
                            if current_timestamp > self.last_processed_timestamp:
                                self.last_processed_timestamp = current_timestamp

                                capture_timestamp = frame_data.get('camera_capture_timestamp')
                                if capture_timestamp is not None:
                                    # self.timestamp_buffer.append(capture_timestamp)
                                    if self.last_capture_timestamp is not None:
                                        time_delta = capture_timestamp - self.last_capture_timestamp
                                        if time_delta > 50:
                                            cprint(f"[{self.name}] Frame drop detected! Gap: {time_delta:.2f} ms", "red", attrs=["bold"])
                                    self.last_capture_timestamp = capture_timestamp

                                rgb_frame = frame_data['image']
                                video_frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
                                video_frame.pts = self.frame_index
                                for packet in self.stream.encode(video_frame):
                                    self.container.mux(packet)
                                self.frame_index += 1
                                self.n_frames_recorded += 1
                    except queue.Empty:
                        time.sleep(0.001)
                    except Exception as e:
                        cprint(f"[{thread_name}] Error processing frame: {e}", "red", attrs=["bold"])
                else:
                    time.sleep(0.001)
                elapsed = mono_time.now_s() - start_time
                sleep_time = max(0, 0.01-elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self._stop_recording_internal()
            cprint(f"[{thread_name}] VideoRecorder process shutting down", "white", attrs=["bold"])
    
    def _start_recording_internal(self, video_path: str, start_time: float):
        if self.is_recording:
            self._stop_recording_internal()
        
        try:
            cprint(f"[{self.name}] Opening video container: {video_path}", "white", attrs=["bold"])
            self.container = av.open(video_path, mode='w')
            self.stream = self.container.add_stream(self.codec, rate=self.fps)

            image_spec = next(spec for spec in self.rgb_queue.array_specs if spec.name == 'image')

            self.stream.width = image_spec.shape[1]
            self.stream.height = image_spec.shape[0]
            self.stream.pix_fmt = self.pixel_format
            self.frame_index = 0
            self.n_frames_recorded = 0
            self.last_capture_timestamp = None # 리셋
            self.recording_start_time = start_time
            self.is_recording=True
            self.timestamp_buffer.clear()
            self.csv_path = pathlib.Path(video_path).with_suffix('.csv')
        except Exception as e:
            cprint(f"[{self.name}] Failed to start recording: {e}", "red", attrs=["bold"])
            self.container = None
            self.stream = None
            self.is_recording = False
    
    def _stop_recording_internal(self):
        if not self.is_recording:
            return

        try:
            # Drain any lingering frames in the queue before closing
            drain_start_time = mono_time.now_s()
            while (mono_time.now_s() - drain_start_time) < 0.1: # Drain for up to 100ms
                try:
                    frame_data = self.rgb_queue.get_nowait()
                    current_timestamp = frame_data.get('timestamp', -1.0)
                    if current_timestamp >= self.recording_start_time and current_timestamp > self.last_processed_timestamp:
                        self.last_processed_timestamp = current_timestamp

                        # 타임스탬프 수집 및 경고 출력 로직 (Final flush에서도 동일하게 수행)
                        capture_timestamp = frame_data.get('camera_capture_timestamp')
                        if capture_timestamp is not None:
                            # self.timestamp_buffer.append(capture_timestamp)
                            if self.last_capture_timestamp is not None:
                                time_delta = capture_timestamp - self.last_capture_timestamp
                                if time_delta > 50:
                                    cprint(f"[{self.name}] Frame drop detected! Gap: {time_delta:.2f} ms", "red", attrs=["bold"])
                            self.last_capture_timestamp = capture_timestamp

                        rgb_frame = frame_data['image']
                        video_frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
                        video_frame.pts = self.frame_index
                        for packet in self.stream.encode(video_frame):
                            self.container.mux(packet)
                        self.frame_index += 1
                        self.n_frames_recorded += 1
                except queue.Empty:
                    break
                except Exception:
                    break

            cprint(f"[{self.name}] Closing video container", "white", attrs=["bold"])
            for packet in self.stream.encode():
                self.container.mux(packet)
            
            self.container.close()
            cprint(f"[{self.name}] Episode finalized with {self.n_frames_recorded} video frames", "cyan", attrs=["bold"])


        except Exception as e:
            cprint(f"[{self.name}] Error closing video container: {e}", "red", attrs=["bold"])
        finally:
            self.container=None
            self.stream=None
            self.is_recording=False
            self.csv_path = None