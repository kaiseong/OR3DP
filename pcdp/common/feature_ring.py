# pcdp/common/feature_ring.py
from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional, Tuple

import torch
from torch import Tensor


class FeatureRing:
    """
    A tiny, robust ring buffer for caching past latent features z_t so you can
    *reuse* them at the next step without re-encoding raw observations.

    - Push the current feature z_t each step (shape [B, D]).
    - Get a fixed-length history stack of the last `n_obs` features.
    - During warm-up (len < n_obs), left-pad with the earliest feature ("edge" pad),
      or zeros if you set pad_mode='zero'.

    Typical use:
        ring = FeatureRing(n_obs=3)
        for t in range(T):
            z_t = enc(P_t)             # [B, D]  (current frame only)
            ring.push(z_t)
            z_hist = ring.stack()      # [B, n_obs*D]
            a_t = policy(z_hist, ...)

    Notes
    -----
    - Keeps *detached* copies to avoid autograd graph bloat.
    - Enforces constant batch size B across pushes.
    - Works on any device; follows the device/dtype of the incoming tensors.
    """

    def __init__(
        self,
        n_obs: int,
        pad_mode: str = "edge",  # 'edge' | 'zero'
        flatten: bool = True,    # True -> [B, n_obs*D], False -> [B, n_obs, D]
    ) -> None:
        if n_obs <= 0:
            raise ValueError("n_obs must be a positive integer.")
        if pad_mode not in ("edge", "zero"):
            raise ValueError("pad_mode must be one of {'edge','zero'}.")

        self.n_obs: int = int(n_obs)
        self.pad_mode: str = pad_mode
        self.flatten: bool = flatten

        self._buf: Deque[Tensor] = deque(maxlen=self.n_obs)
        self._B: Optional[int] = None
        self._D: Optional[int] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

    # ------------------------------ core API ------------------------------

    @torch.no_grad()
    def push(self, z_t: Tensor) -> None:
        """
        Add the current latent feature to the ring.

        Args:
            z_t: Tensor of shape [B, D]. Will be detached and stored.
        """
        if z_t.dim() != 2:
            raise ValueError(f"z_t must be [B, D], got {tuple(z_t.shape)}")

        B, D = z_t.shape
        if self._B is None:
            # Initialize shape/device traits on first push
            self._B, self._D = B, D
            self._device, self._dtype = z_t.device, z_t.dtype
        else:
            if B != self._B:
                raise ValueError(f"Batch size changed: {B} != {self._B}")
            if D != self._D:
                raise ValueError(f"Feature dim changed: {D} != {self._D}")

        self._buf.append(z_t.detach())

    @torch.no_grad()
    def extend(self, zs: Iterable[Tensor]) -> None:
        """Push multiple frames in order (each [B, D])."""
        for z in zs:
            self.push(z)

    @torch.no_grad()
    def stack(self) -> Tensor:
        """
        Return a history stack of length `n_obs`.

        Returns:
            Tensor with shape:
              - [B, n_obs*D] if flatten=True
              - [B, n_obs, D] if flatten=False
        """
        if self._B is None or self._D is None or len(self._buf) == 0:
            raise RuntimeError("FeatureRing is empty. Call push(z_t) at least once.")

        # Build list with left-padding
        frames = list(self._buf)
        if len(frames) < self.n_obs:
            if self.pad_mode == "edge":
                first = frames[0]
                pad_cnt = self.n_obs - len(frames)
                frames = [first] * pad_cnt + frames
            else:  # 'zero'
                zeros = torch.zeros(self._B, self._D, device=self._device, dtype=self._dtype)
                pad_cnt = self.n_obs - len(frames)
                frames = [zeros] * pad_cnt + frames
        else:
            # already at capacity
            pass

        hist = torch.stack(frames, dim=1)  # [B, n_obs, D]
        return hist.reshape(self._B, self.n_obs * self._D) if self.flatten else hist

    # ------------------------------ utilities -----------------------------

    @torch.no_grad()
    def last(self) -> Tensor:
        """Return the most recent feature [B, D]."""
        if len(self._buf) == 0:
            raise RuntimeError("FeatureRing is empty.")
        return self._buf[-1]

    def __len__(self) -> int:
        """Number of frames currently stored (<= n_obs)."""
        return len(self._buf)

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        """Return (B, D) after first push, else None."""
        return None if self._B is None else (self._B, self._D)  # type: ignore[return-value]

    def reset(self) -> None:
        """Clear the ring (e.g., at episode boundaries)."""
        self._buf.clear()
        self._B = self._D = None
        self._device = self._dtype = None

    def is_warm(self) -> bool:
        """True if the ring already holds n_obs frames (i.e., no padding needed)."""
        return len(self._buf) == self.n_obs
