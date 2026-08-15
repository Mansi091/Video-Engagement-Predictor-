"""Extract video frame features using CLIP (ViT-B/32)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def extract_frames(
    video_path: Path,
    frames_per_video: int = 8,
    frame_interval_sec: float = 2.0,
) -> list[Image.Image]:
    """Sample frames from MP4 using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_step = max(int(fps * frame_interval_sec), 1)
    frames = []
    frame_idx = 0

    while len(frames) < frames_per_video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        frame_idx += frame_step

    cap.release()
    return frames


class CLIPVideoEncoder:
    """Encode video clips into fixed-size embeddings via CLIP frame pooling."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        batch_size: int = 8,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def encode_frames(self, frames: list[Image.Image]) -> np.ndarray:
        """Mean-pool CLIP image embeddings across frames."""
        if not frames:
            return np.zeros(512, dtype=np.float32)

        self._load()
        embeddings = []

        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

        all_emb = np.vstack(embeddings)
        return all_emb.mean(axis=0).astype(np.float32)

    def encode_video(
        self,
        video_path: Path,
        frames_per_video: int = 8,
        frame_interval_sec: float = 2.0,
    ) -> np.ndarray:
        frames = extract_frames(video_path, frames_per_video, frame_interval_sec)
        return self.encode_frames(frames)

    def encode_all_clips(
        self,
        video_dir: Path,
        frames_per_video: int = 8,
        frame_interval_sec: float = 2.0,
    ) -> dict[int, np.ndarray]:
        """Encode each MP4 in video_dir; key = sorted index 0..N-1."""
        video_dir = Path(video_dir)
        mp4s = sorted(video_dir.glob("*.mp4"))
        result = {}
        for i, path in enumerate(mp4s):
            print(f"CLIP encoding: {path.name}")
            result[i] = self.encode_video(path, frames_per_video, frame_interval_sec)
        return result
