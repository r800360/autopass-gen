"""Save CARLA ego + overhead frames and stitch MP4 for demos."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


def _draw_hud(rgb: np.ndarray, lines: List[str]) -> np.ndarray:
    if Image is None:
        return rgb
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    y = 8
    for line in lines:
        draw.rectangle([4, y - 2, 4 + 8 * len(line), y + 14], fill=(0, 0, 0))
        draw.text((8, y), line, fill=(255, 255, 0))
        y += 18
    return np.array(img)


def _stack_views(ego: np.ndarray, overhead: Optional[np.ndarray]) -> np.ndarray:
    """Side-by-side ego camera + bird's-eye spectator for comprehensive footage."""
    if overhead is None or overhead.shape[0] == 0:
        return ego
    eh, ew = ego.shape[:2]
    oh, ow = overhead.shape[:2]
    target_h = eh
    scale = target_h / max(1, oh)
    ow2 = int(ow * scale)
    if Image is not None:
        over = np.array(Image.fromarray(overhead).resize((ow2, target_h)))
    else:
        over = overhead[:, :ow2] if ow2 <= ow else overhead
    pad = np.zeros((eh, 4, 3), dtype=np.uint8)
    return np.concatenate([ego, pad, over[:, : min(ow2, ew)]], axis=1)


class CarlaRecorder:
    def __init__(self, out_dir: Path, scenario_id: str) -> None:
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / "frames" / scenario_id
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: List[dict] = []
        self._index = 0
        self._display_t_s = 0.0
        self._last_graph_t_s = -1.0

    def next_display_t_s(self, graph_t_s: float, *, label: str) -> float:
        """
        Monotonic HUD time for demo frames.

        With AUTOPASS_VIDEO_REALTIME=1 (hero-pass demos), HUD time tracks sim time only
        so the video never advances while CARLA is frozen between executes.
        """
        import os

        gt = float(graph_t_s)
        if gt > self._last_graph_t_s + 1e-6:
            self._display_t_s = gt
            self._last_graph_t_s = gt
            return self._display_t_s
        label_u = (label or "").upper()
        if label_u.startswith("EXECUTE") or label_u.startswith("DONE"):
            self._display_t_s = max(self._display_t_s, gt)
            self._last_graph_t_s = max(self._last_graph_t_s, gt)
            return self._display_t_s
        if os.environ.get("AUTOPASS_VIDEO_REALTIME", "").strip() in ("1", "true", "yes"):
            return max(self._display_t_s, gt)
        step_s = 0.05
        if label_u in ("PLANNER", "RUN_TOOL", "CRITIQUE_MANEUVER", "CRITIQUE_TOOL"):
            step_s = 0.04
        self._display_t_s = max(self._display_t_s + step_s, gt)
        return self._display_t_s

    def capture(
        self,
        rgb: np.ndarray,
        *,
        t_s: float,
        label: str,
        extra: Optional[dict] = None,
        overhead: Optional[np.ndarray] = None,
        graph_step: Optional[int] = None,
    ) -> None:
        lines = [f"t={t_s:.2f}s", f"frame={self._index}", label]
        if extra:
            graph_t = extra.get("graph_t_s")
            if graph_t is not None and abs(float(graph_t) - float(t_s)) > 0.02:
                lines.append(f"sim_t={float(graph_t):.2f}s")
        if graph_step is not None:
            lines.append(f"step={graph_step}")
        if extra:
            for k, v in list(extra.items())[:5]:
                lines.append(f"{k}: {v}")
        hud = _draw_hud(rgb, lines)
        composite = _stack_views(hud, overhead)
        path = self.frames_dir / f"frame_{self._index:05d}.png"
        Image.fromarray(composite).save(path)
        meta = {
            "file": path.name,
            "t_s": t_s,
            "graph_t_s": extra.get("graph_t_s") if extra else None,
            "label": label,
            "extra": extra or {},
        }
        if graph_step is not None:
            meta["graph_step"] = graph_step
        self.metadata.append(meta)
        self._index += 1

    def _video_fps(self, frame_count: int, default_fps: int = 12) -> int:
        """Match playback rate to recorded HUD sim-time span (avoids frozen or fast-forward video)."""
        if frame_count < 2 or not self.metadata:
            return default_fps
        try:
            t0 = float(self.metadata[0].get("t_s", 0.0))
            t1 = float(self.metadata[-1].get("t_s", t0))
        except (TypeError, ValueError):
            return default_fps
        span = max(0.05, t1 - t0)
        derived = int(round((frame_count - 1) / span))
        return max(20, min(24, derived))

    def write_video(self, name: str = "run.mp4", fps: int | None = None) -> Optional[Path]:
        if self._index == 0:
            return None
        try:
            import imageio.v3 as iio
        except ImportError:
            print("[CARLA] Install imageio for MP4: pip install imageio imageio-ffmpeg")
            return None
        paths = sorted(self.frames_dir.glob("frame_*.png"))
        if fps is None:
            fps = self._video_fps(len(paths))
        frames = [iio.imread(p) for p in paths]
        out = self.out_dir / name
        iio.imwrite(out, frames, fps=fps, codec="libx264")
        (self.out_dir / "frames_metadata.json").write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )
        print(f"[CARLA] Video: {out} ({len(frames)} frames @ {fps} fps)")
        return out

    def write_split_videos(
        self,
        ego_name: str,
        overhead_name: str,
        *,
        fps: int = 8,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Write separate ego (left crop) and overhead (right crop) MP4s from composite frames."""
        if self._index == 0 or Image is None:
            return None, None
        try:
            import imageio.v3 as iio
        except ImportError:
            return None, None
        paths = sorted(self.frames_dir.glob("frame_*.png"))
        if not paths:
            return None, None
        sample = iio.imread(paths[0])
        h, w = sample.shape[:2]
        mid = w // 2
        ego_frames = [iio.imread(p)[:, :mid] for p in paths]
        over_frames = [iio.imread(p)[:, mid:] for p in paths]
        ego_out = self.out_dir / ego_name
        over_out = self.out_dir / overhead_name
        iio.imwrite(ego_out, ego_frames, fps=fps, codec="libx264")
        iio.imwrite(over_out, over_frames, fps=fps, codec="libx264")
        return ego_out, over_out
