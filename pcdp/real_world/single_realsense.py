# single_orbbec.py
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
import pcdp.common.mono_time as mono_time
from termcolor import cprint
import pyrealsense2 as rs
from pcdp.shared_memory.shared_ndarray import SharedNDArray
from pcdp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


def rs_points_to_array(points: rs.points,
                       color_frame: rs.video_frame,
                       min_z: float = 0.07,
                       max_z: float | None = 0.50,
                       bilinear: bool = True) -> np.ndarray:
    """points + color -> (N,6) float32 [x,y,z,r,g,b], RGB in [0,1]."""
    # 1) zero-copy view
    xyz = np.frombuffer(points.get_vertices(), dtype=np.float32).reshape(-1, 3)
    uv  = np.frombuffer(points.get_texture_coordinates(), dtype=np.float32).reshape(-1, 2)

    color = np.asanyarray(color_frame.get_data())  # HxWx3 BGR uint8
    H, W, _ = color.shape

    # 2) 유효 마스크
    mask = np.isfinite(xyz).all(axis=1)
    mask &= (xyz[:, 2] > min_z)
    if max_z is not None:
        mask &= (xyz[:, 2] < max_z)
    mask &= (uv[:, 0] >= 0) & (uv[:, 0] < 1) & (uv[:, 1] >= 0) & (uv[:, 1] < 1)
    if not np.any(mask):
        return np.empty((0, 6), dtype=np.float32)

    xyz = xyz[mask]
    uv  = uv[mask]

    # 3) 텍스처 샘플링
    if not bilinear:
        u = (uv[:, 0] * (W - 1)).astype(np.int32)
        v = (uv[:, 1] * (H - 1)).astype(np.int32)
        bgr = color[v, u].astype(np.float32)
    else:
        u  = uv[:, 0] * (W - 1)
        v  = uv[:, 1] * (H - 1)
        u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
        u1 = np.clip(u0 + 1, 0, W - 1);    v1 = np.clip(v0 + 1, 0, H - 1)
        du = (u - u0)[..., None];          dv = (v - v0)[..., None]

        c00 = color[v0, u0].astype(np.float32)
        c10 = color[v0, u1].astype(np.float32)
        c01 = color[v1, u0].astype(np.float32)
        c11 = color[v1, u1].astype(np.float32)
        bgr = (c00 * (1 - du) * (1 - dv) +
               c10 * (    du) * (1 - dv) +
               c01 * (1 - du) * (    dv) +
               c11 * (    du) * (    dv))

    rgb = bgr[..., ::-1] / 255.0  # BGR->RGB, [0,1]
    return np.concatenate([xyz.astype(np.float32), rgb.astype(np.float32)], axis=1)

class SingleRealSense(mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        resolution=(424, 240),
        put_fps=60,
        get_max_k=60,
        verbose=False,
        # Depth filter parameters (applied during data collection)
        # Threshold + Spatial filters are applied and SAVED to depth_image
        # Voxel downsampling is applied later during training/inference
        enable_threshold_filter=True,
        threshold_min_dist=0.04,  # meter
        threshold_max_dist=0.20,  # meter
        enable_spatial_filter=True,
        spatial_magnitude=2,      # 반복 횟수 (1~5)
        spatial_smooth_alpha=0.7, # 필터 강도 (0.25~1.0)
        spatial_smooth_delta=15   # Edge 감지 임계값 (1~50)
    ):
        super().__init__()

        # create ring buffer
        examples = dict()
        # pointcloud field kept for backward compatibility but not used (fixed small size)
        examples['pointcloud'] = np.empty(shape=(1, 6), dtype=np.float32)
        examples['depth_image'] = np.empty(shape=resolution[::-1], dtype=np.uint16)
        examples['color_image'] = np.empty(shape=resolution[::-1] + (3,), dtype=np.uint8)
        examples['intrinsics'] = np.empty(shape=(4,), dtype=np.float32)  # [fx, fy, cx, cy]
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=get_max_k,
            get_time_budget=0.05,
            put_desired_frequency=put_fps
        )
        
        shape = resolution[::-1]
        examples_ = dict()

        examples_['image'] = np.empty(
            shape=(*shape, 3), dtype=np.uint8)
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

        # create shared array for intrinsics + depth_scale
        # [0:4] = fx, fy, cx, cy
        # [4:6] = height, width
        # [6] = depth_scale
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager,
            shape=(7,),
            dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # copied variables
        self.resolution = resolution
        self.put_fps = put_fps
        self.verbose = verbose
        self.put_start_time = None

        # filter parameters
        self.enable_threshold_filter = enable_threshold_filter
        self.threshold_min_dist = threshold_min_dist
        self.threshold_max_dist = threshold_max_dist
        self.enable_spatial_filter = enable_spatial_filter
        self.spatial_magnitude = spatial_magnitude
        self.spatial_smooth_alpha = spatial_smooth_alpha
        self.spatial_smooth_delta = spatial_smooth_delta

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.rgb_ring_buffer = rgb_ring_buffer
        self.intrinsics_array = intrinsics_array

    @staticmethod
    def get_connected_devices():
        serials = list()
        try:
            for d in rs.context().devices:
                if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                    serial = d.get_info(rs.camera_info.serial_number)
                    product_line = d.get_info(rs.camera_info.product_line)
                    if product_line == 'D400':
                        # only works with D400 series
                        serials.append(serial)
        except Exception:
            pass
        serials = sorted(serials)
        return serials

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
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, cx, cy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = cx
        mat[1,2] = cy
        return mat
    
    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale
    
    # ========= internal API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)

        pipeline = rs.pipeline()
        cfg = rs.config()
        
        w, h = self.resolution
        fps = self.put_fps

        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        profile = pipeline.start(cfg)

        # Get depth sensor and depth_scale
        depth_sensor = None
        depth_scale = 0.001  # Default fallback
        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            if self.verbose:
                cprint(f"[SingleRealSense] Depth scale: {depth_scale}", "cyan", attrs=["bold"])
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 4.0)
        except Exception as e:
            cprint(f"[SingleRealSense] visual_preset not supported: {e}", "yellow", attrs=["bold"])
            pass

        # Setup post-processing filters
        threshold_filter = None
        spatial_filter = None

        if self.enable_threshold_filter:
            threshold_filter = rs.threshold_filter()
            threshold_filter.set_option(rs.option.min_distance, self.threshold_min_dist)
            threshold_filter.set_option(rs.option.max_distance, self.threshold_max_dist)
            if self.verbose:
                cprint(f"[SingleRealSense] Threshold filter enabled: {self.threshold_min_dist}m ~ {self.threshold_max_dist}m", "cyan", attrs=["bold"])

        if self.enable_spatial_filter:
            spatial_filter = rs.spatial_filter()
            spatial_filter.set_option(rs.option.filter_magnitude, self.spatial_magnitude)
            spatial_filter.set_option(rs.option.filter_smooth_alpha, self.spatial_smooth_alpha)
            spatial_filter.set_option(rs.option.filter_smooth_delta, self.spatial_smooth_delta)
            if self.verbose:
                cprint(f"[SingleRealSense] Spatial filter enabled: mag={self.spatial_magnitude}, alpha={self.spatial_smooth_alpha}, delta={self.spatial_smooth_delta}", "cyan", attrs=["bold"])

        # warmup
        for _ in range(30):
            pipeline.wait_for_frames()
        pc = rs.pointcloud()
        
        pre_time = 0
        anormaly_cnt=0
        
        try:
            active_profile = pipeline.get_active_profile()
            color_profile = rs.video_stream_profile(active_profile.get_stream(rs.stream.color))
            color_intr = color_profile.get_intrinsics()

            self.intrinsics_array.get()[0] = color_intr.fx
            self.intrinsics_array.get()[1] = color_intr.fy
            self.intrinsics_array.get()[2] = color_intr.ppx
            self.intrinsics_array.get()[3] = color_intr.ppy
            self.intrinsics_array.get()[4] = color_intr.height
            self.intrinsics_array.get()[5] = color_intr.width
            self.intrinsics_array.get()[6] = depth_scale

            if self.verbose:
                cprint(f"[SingleRealSense] Initialized with resolution {w}x{h} at {fps} FPS.", "green", attrs=["bold"])
                cprint(f"[SingleRealSense] Main loop started.", "green", attrs=["bold"])

            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = mono_time.now_s()
            
            iter_idx = 0

            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue

                # Apply SDK filters in order: Threshold → Spatial
                # These filters are applied and SAVED to depth_image
                if threshold_filter is not None:
                    depth = threshold_filter.process(depth)

                if spatial_filter is not None:
                    depth = spatial_filter.process(depth)

                # Get filtered depth array (after SDK filters)
                depth_array_filtered = np.asanyarray(depth.get_data(), dtype=np.uint16)

                if color:
                    rgb_array = np.frombuffer(color.get_data(), dtype=np.uint8).reshape((h, w, 3))
                else:
                    rgb_array = np.zeros((h,w, 3), dtype=np.uint8)

                # Get intrinsics
                instrinsics_data = self.intrinsics_array.get()[:4]  # [fx, fy, cx, cy]

                receive_time = mono_time.now_s()

                # Pointcloud generation removed - now using depth_image + OpenCV filters
                # Keeping minimal pointcloud for backward compatibility
                final_pc = np.zeros((1, 6), dtype=np.float32)

                depth_time = depth.get_timestamp()
                rgb_time = color.get_timestamp()

                if (depth_time - pre_time) > (1/self.put_fps)*1000+5:
                    anormaly_cnt+=1
                pre_time=depth_time

                data = dict()
                data['camera_receive_timestamp'] = receive_time
                data['camera_capture_timestamp'] = depth_time
                data['pointcloud']=final_pc
                data['depth_image'] = depth_array_filtered  # Store SDK filtered depth (threshold + spatial)
                data['color_image'] = rgb_array[:, :, ::-1]  # Store color image (BGR→RGB)
                data['intrinsics'] = instrinsics_data  # Add intrinsics

                data_ = dict()
                data_['camera_receive_timestamp'] = receive_time
                data_['camera_capture_timestamp'] = rgb_time
                data_['image'] = rgb_array[:, :, ::-1] # BGR->RGB


                put_data = data
                put_data_ = data_
                
                put_data['timestamp'] = receive_time
                put_data_['timestamp'] = receive_time
                try:
                    self.ring_buffer.put(put_data, wait=False)
                    self.rgb_ring_buffer.put(put_data_, wait=False)
                except TimeoutError:
                        if self.verbose:
                            cprint("[SingleRealSense] Ring buffer full, dropping frame.", "yellow", attrs=["bold"])
                        continue
                        
                if iter_idx ==0:
                    self.ready_event.set()
                
                iter_idx += 1
                
                
        except Exception as e:
            with open("/tmp/realsense_error.log", "w") as f:
                f.write(f"Error in SingleRealSense main loop: {e}\n")
                import traceback
                traceback.print_exc(file=f)
            cprint(f"[SingleRealSense] Error in main loop: {e}", "red", attrs=["bold"])
            import traceback
            traceback.print_exc()
        finally:
            pipeline.stop()
            # self.ready_event.set()
            cprint(f"[Realsense] anormaly_cnt: {anormaly_cnt}", "red", attrs=["bold"])
            if self.verbose:
                cprint("[SingleRealSense] Main loop ended, resources cleaned up.", "red", attrs=["bold"])