# single_orbbec.py
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
import pcdp.common.mono_time as mono_time
from pcdp.common.orbbec_util import frame_to_rgb_frame
from termcolor import cprint

import pyorbbecsdk as ob
from pcdp.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from pcdp.shared_memory.shared_ndarray import SharedNDArray
from pcdp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

# debug
import csv
def save_timestamp_duration_to_csv(timestamps, filename):
    """타임스탬프 배열을 CSV 파일로 저장"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'timestamp'])
        for i, ts in enumerate(timestamps):
            writer.writerow([i, ts])

class SingleOrbbec(mp.Process):
    MAX_PATH_LENGTH = 4096

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        rgb_resolution=(1280, 720),
        put_fps=30,
        put_downsample=False,
        get_max_k=30,
        mode="C2D",
        verbose=False
    ):
        super().__init__()
        
        
        # create ring buffer
        examples = dict()
        
        if mode == "D2C":
            examples['pointcloud'] = np.empty(
                shape=(921600, 6), dtype=np.float32)  # XYZ + RGB
        elif mode == "C2D":
            examples['pointcloud'] = np.empty(
                shape=(92160, 6), dtype=np.float32) # 368640 = 640*576  92160 = 320*288
        else:
            raise RuntimeError("mode is wrong")
        
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=get_max_k,
            get_time_budget=0.025,
            put_desired_frequency=put_fps
        )
        
        rgb_resolution=rgb_resolution[::-1]
        examples_ = dict()
        examples_['image'] = np.empty(
            shape=(*rgb_resolution, 3), dtype=np.uint8)
        examples_['camera_capture_timestamp'] = 0.0
        examples_['camera_receive_timestamp'] = 0.0
        examples_['timestamp'] = 0.0
        examples_['step_idx'] = 0
        
        rgb_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples_,
            get_max_k=get_max_k,
            get_time_budget=0.05,
            put_desired_frequency=put_fps
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager,
            shape=(6,), 
            dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # copied variables
        self.resolution = rgb_resolution
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.mode = mode
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.rgb_ring_buffer = rgb_ring_buffer
        self.intrinsics_array = intrinsics_array

    @staticmethod
    def get_connected_devices():
        """Get list of connected Orbbec devices"""
        if ob is None:
            return []
        
        try:
            ctx = ob.Context()
            devices = ctx.query_devices()
            device_list = []
            for i in range(devices.get_count()):
                device = devices.get_device_by_index(i)
                device_info = device.get_device_info()
                device_list.append({
                    'index': i,
                    'name': device_info.get_name(),
                    'serial': device_info.get_serial_number(),
                    'pid': device_info.get_pid(),
                    'vid': device_info.get_vid()
                })
            return device_list
        except Exception as e:
            print(f"Error querying Orbbec devices: {e}")
            return []

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = mono_time.now_s()
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_rgb(self, out=None):
        return self.rgb_ring_buffer.get(out=out)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, cx, cy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = cx
        mat[1,2] = cy
        return mat

    
    # ========= internal API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        
        pipeline = ob.Pipeline()
        cfg = ob.Config()

        depth_res = (320, 288)
        color_res = self.resolution[::-1]
        
        if self.mode=="D2C":
            align_to = ob.OBStreamType.COLOR_STREAM
        elif self.mode=="C2D":
            align_to = ob.OBStreamType.DEPTH_STREAM
        
        w, h = self.resolution
        fps = self.put_fps

        depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)\
                        .get_video_stream_profile(*depth_res,ob.OBFormat.Y16,fps)
        color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)\
                        .get_video_stream_profile(*color_res,ob.OBFormat.RGB,fps)
        cfg.enable_stream(depth_profile)
        cfg.enable_stream(color_profile)
        cfg.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipeline.enable_frame_sync()
        pipeline.start(cfg)
        align = ob.AlignFilter(align_to_stream=align_to)
        pc_filter = ob.PointCloudFilter()
        cam_param = pipeline.get_camera_param()
        pc_filter.set_camera_param(cam_param)
        pc_filter.set_create_point_format(ob.OBFormat.RGB_POINT)
        
        pre_time = 0
        anormaly_cnt=0

        try:
            if self.mode == "D2C":
                intr = cam_param.rgb_intrinsic
            elif self.mode == "C2D":
                intr = cam_param.depth_intrinsic

            self.intrinsics_array.get()[0] = intr.fx
            self.intrinsics_array.get()[1] = intr.fy
            self.intrinsics_array.get()[2] = intr.cx
            self.intrinsics_array.get()[3] = intr.cy
            self.intrinsics_array.get()[4] = intr.height
            self.intrinsics_array.get()[5] = intr.width

            if self.verbose:
                cprint(f"[SingleOrbbec] Initialized with resolution {w}x{h} at {fps} FPS.", "green", attrs=["bold"])
                cprint(f"[SingleOrbbec] Main loop started.", "green", attrs=["bold"])

            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = mono_time.now_s()
            
            iter_idx =0


            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(1)
                if frames is None:
                    continue
                
                depth, color = frames.get_depth_frame(), frames.get_color_frame()
                if depth is None or color is None:
                    continue
                
                # rgb_image shape is (720, 1280, 3)
                # rgb_image = frame_to_rgb_frame(color)

                if color:
                    img_width = color.get_width()
                    img_height = color.get_height()
                    rgb_array = np.frombuffer(color.get_data(), dtype=np.uint8).reshape((*self.resolution, 3))
                else:
                    rgb_array = np.zeros((*self.resolution, 3), dtype=np.uint8)
                
                receive_time = mono_time.now_s()

                frame = align.process(frames)
                pc_filter.set_position_data_scaled(depth.get_depth_scale())
                point_cloud = pc_filter.calculate(pc_filter.process(frame))  # (N,6) float32

                
                depth_time = depth.get_timestamp()
                rgb_time = color.get_timestamp()
                if (depth_time - pre_time) > 35:
                    anormaly_cnt+=1
                pre_time=depth_time


                if point_cloud is not None:
                    points_data = np.asarray(point_cloud, dtype=np.float32)
                    if len(points_data.shape) == 1:
                        points_data = points_data.reshape(-1, 6)
                    points_data[:, 3:] = points_data[:, 3:] / 255.0  # RGB [0,1]
                else:
                    points_data = np.empty((0, 6), dtype=np.float32)


                data = dict()
                data['camera_receive_timestamp'] = receive_time
                data['camera_capture_timestamp'] = depth_time
                data['pointcloud']=points_data

                data_ = dict()
                data_['camera_receive_timestamp'] = receive_time
                data_['camera_capture_timestamp'] = rgb_time
                data_['image'] = rgb_array


                put_data = data
                put_data_ = data_

                if self.put_downsample:
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt =1/self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = receive_time
                        try:
                            self.ring_buffer.put(put_data, wait=False)
                        except TimeoutError:
                            if self.verbose:
                                cprint("[SingleOrbbec] Ring buffer full, dropping frame.", "green", attrs=["bold"])
                            break
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    put_data_['step_idx'] = step_idx
                    put_data_['timestamp'] = receive_time
                    try:
                        self.ring_buffer.put(put_data, wait=False)
                        self.rgb_ring_buffer.put(put_data_, wait=False)
                    except TimeoutError:
                            if self.verbose:
                                cprint("[SingleOrbbec] Ring buffer full, dropping frame.", "green", attrs=["bold"])
                            continue
                
                if iter_idx ==0:
                    self.ready_event.set()
                
                
                iter_idx += 1
                
                
        except Exception as e:
            with open("/tmp/orbbec_error.log", "w") as f:
                f.write(f"Error in SingleOrbbec main loop: {e}\n")
                import traceback
                traceback.print_exc(file=f)
            cprint(f"[SingleOrbbec] Error in main loop: {e}", "green", attrs=["bold"])
            import traceback
            traceback.print_exc()
        finally:
            pipeline.stop()
            # self.ready_event.set()
            cprint(f"anormaly_cnt: {anormaly_cnt}", "green", attrs=["bold"])
            if self.verbose:
                cprint("[SingleOrbbec] Main loop ended, resources cleaned up.", "green", attrs=["bold"])




