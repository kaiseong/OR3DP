import time

_MONO0_NS  = time.monotonic_ns()
_WALL0_NS  = time.time_ns()

def now_ns() -> int:          
    return time.monotonic_ns()

def now_ms() -> float:
    return time.monotonic_ns() / 1e6

def now_s()  -> float:        
    return now_ns() / 1e9

def wall_from_mono(mono_ns: int) -> float:
    "단조시(ns) → 벽시계(s)"
    return (mono_ns - _MONO0_NS + _WALL0_NS) / 1e9

def mono_from_wall(wall_s: float) -> float:
    return now_s() + (wall_s - time.time())