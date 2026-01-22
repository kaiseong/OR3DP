#!/usr/bin/env python3
"""
Point Cloud Pipeline Debug Utilities
Unified debug output for pc_visualize.py and eval_piper_RISE.py
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineDebugInfo:
    """Container for pipeline debug information."""
    # Point counts
    n_femto: int = 0
    n_d405_raw: int = 0
    n_d405_voxel: int = 0
    n_blended: int = 0
    n_variance: int = 0  # variance filter output
    n_transform: int = 0
    n_crop: int = 0
    n_ror: int = 0

    # Timing (ms)
    t_femto: float = 0.0
    t_d405: float = 0.0
    t_voxel: float = 0.0
    t_blend: float = 0.0
    t_variance: float = 0.0  # variance filter time
    t_transform: float = 0.0
    t_crop: float = 0.0
    t_ror: float = 0.0

    # Process/Viz/Total timing
    t_process: float = 0.0
    t_viz: float = 0.0
    t_total: float = 0.0

    # Flags (enabled/disabled)
    enable_d405: bool = True
    enable_voxel: bool = True
    enable_blending: bool = True
    enable_variance: bool = False  # variance filter
    enable_crop: bool = True
    enable_ror: bool = False

    # Frame count
    frame_count: int = 0


def format_pipeline_debug(info: PipelineDebugInfo) -> str:
    """
    Format pipeline debug info into a readable string.

    Args:
        info: PipelineDebugInfo object

    Returns:
        Formatted debug string
    """
    lines = []
    lines.append(f"\n[Frame {info.frame_count}] Point Cloud Pipeline Debug")
    lines.append("=" * 80)

    # 1. Femto Bolt (always enabled)
    lines.append(f"  1. get femto bolt pt:    {info.n_femto:>10,}  points     time: {info.t_femto:>6.1f} ms")

    # 2. D405 (+filter)
    if info.enable_d405:
        lines.append(f"  2. get d405 pt(+filter): {info.n_d405_raw:>10,}  points     time: {info.t_d405:>6.1f} ms")
    else:
        lines.append(f"  2. get d405 pt(+filter):          -  points     time:      X")

    # 3. D405 voxel
    if info.enable_voxel and info.enable_d405:
        lines.append(f"  3. D405 voxel:           {info.n_d405_voxel:>10,}  points     time: {info.t_voxel:>6.1f} ms")
    else:
        lines.append(f"  3. D405 voxel:                    -  points     time:      X")

    # 4. Blending
    if info.enable_blending and info.enable_d405:
        lines.append(f"  4. blending:             {info.n_blended:>10,}  points     time: {info.t_blend:>6.1f} ms")
    else:
        lines.append(f"  4. blending:                      -  points     time:      X")

    # 5. Variance filter
    if info.enable_variance:
        lines.append(f"  5. variance filter:      {info.n_variance:>10,}  points     time: {info.t_variance:>6.1f} ms")
    else:
        lines.append(f"  5. variance filter:               -  points     time:      X")

    # 6. Base transform (always enabled)
    lines.append(f"  6. base transform:       {info.n_transform:>10,}  points     time: {info.t_transform:>6.1f} ms")

    # 7. Workspace crop
    if info.enable_crop:
        lines.append(f"  7. workspace crop:       {info.n_crop:>10,}  points     time: {info.t_crop:>6.1f} ms")
    else:
        lines.append(f"  7. workspace crop:                -  points     time:      X")

    # 8. ROR filter
    if info.enable_ror:
        lines.append(f"  8. ROR filter:           {info.n_ror:>10,}  points     time: {info.t_ror:>6.1f} ms")
    else:
        lines.append(f"  8. ROR filter:                    -  points     time:      X")

    lines.append("=" * 80)

    # Timing summary
    fps = 1000.0 / info.t_total if info.t_total > 0 else 0.0
    lines.append(f"  Timing: Process={info.t_process:.1f}ms, Viz={info.t_viz:.1f}ms, Total={info.t_total:.1f}ms ({fps:.1f} FPS)")

    return "\n".join(lines)


def print_pipeline_debug(info: PipelineDebugInfo):
    """Print pipeline debug info."""
    print(format_pipeline_debug(info))
