from typing import Union
import numbers
import numpy as np
import scipy.spatial.transform as st

def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    # quaternion distance
    rot_dist = (end_rot * start_rot.inv()).magnitude()
    return pos_dist, rot_dist

class PoseTrajectoryInterpolator:
    """
    This class provides a non-interpolating trajectory handler.
    It holds the pose of a waypoint until the time for the next waypoint is reached.
    This is useful for systems where smooth interpolation is not desirable,
    and direct command following is preferred. It correctly handles sequences
    of waypoints for inference.
    """
    def __init__(self, times: np.ndarray, poses: np.ndarray, logger=None):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)
        
        # Sort by time to ensure correctness
        sort_indices = np.argsort(times)
        self._times = times[sort_indices]
        self._poses = poses[sort_indices]
        self.logger = logger

    @property
    def times(self) -> np.ndarray:
        return self._times
    
    @property
    def poses(self) -> np.ndarray:
        return self._poses

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        """
        For a given time t, find the most recent waypoint and return its pose.
        The pose is held constant until the next waypoint's time is reached.
        """
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        # For each time in t, find the index of the waypoint that should be active.
        # 'side="right"' finds the insertion point, so idx-1 is the correct index
        # for the last waypoint whose time is <= t.
        indices = np.searchsorted(self._times, t, side='right')
        # Clamp indices to be at least 1 to avoid index -1. If t is before the
        # first waypoint, it should take the pose of the first waypoint.
        indices = np.maximum(1, indices)
        # The actual pose index is one less than the found index.
        pose_indices = indices - 1
        
        pose = self._poses[pose_indices]

        if is_single:
            pose = pose[0]
        return pose

    def trim(self, 
            start_t: float, end_t: float
            ) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t <= times) & (times <= end_t)
        
        # Also include the last waypoint before the start_t
        before_start_indices = np.where(times < start_t)[0]
        if len(before_start_indices) > 0:
            last_before_idx = before_start_indices[-1]
            should_keep[last_before_idx] = True

        keep_times = times[should_keep]
        keep_poses = self.poses[should_keep]

        # If no waypoints are within the range, use the start and end times
        # and sample the pose.
        if len(keep_times) == 0:
             all_times = np.array([start_t, end_t])
             all_poses = self(all_times)
        else:
            all_times = keep_times
            all_poses = keep_poses

        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)

    def drive_to_waypoint(self, 
            pose, time, curr_time,
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf
        ) -> "PoseTrajectoryInterpolator":
        time = max(time, curr_time)
        
        # Trim interpolator to the current time, keeping only past waypoints
        keep_idxs = np.where(self.times <= curr_time)[0]
        
        # Append the new waypoint
        times = np.append(self.times[keep_idxs], [time])
        poses = np.append(self.poses[keep_idxs], [pose], axis=0)

        # Create a new interpolator with the updated schedule
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(self,
            pose, time, 
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "PoseTrajectoryInterpolator":
        
        if curr_time is not None and time <= curr_time:
            return self

        # Find waypoints that are still relevant.
        # We keep all waypoints that have already passed plus the new one.
        if curr_time is None:
            curr_time = self.times[0] # Default to the beginning

        # Keep waypoints that are in the past relative to the new waypoint's time
        # and in the future relative to the current time.
        keep_idxs = np.where(self.times < time)[0]

        new_times = np.append(self.times[keep_idxs], [time])
        new_poses = np.append(self.poses[keep_idxs], [pose], axis=0)
        
        # Ensure times are unique and sorted, which might happen with aggressive scheduling
        unique_times, unique_indices = np.unique(new_times, return_index=True)
        unique_poses = new_poses[unique_indices]

        final_interp = PoseTrajectoryInterpolator(unique_times, unique_poses, logger=self.logger)
        return final_interp