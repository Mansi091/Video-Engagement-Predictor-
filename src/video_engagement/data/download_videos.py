"""Download sample MP4 clips or generate synthetic fallback videos."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_SAMPLE_URLS = [
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VideoEngagementPipeline/1.0"


def _download_one(url: str, dest: Path) -> bool:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"Warning: failed to download {dest.name}: {e}")
        return False


def generate_synthetic_videos(video_dir: Path, count: int = 10) -> list[Path]:
    """Create short synthetic MP4s (unique colors/motion per clip) when downloads fail."""
    import cv2
    import numpy as np

    video_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = [
        (255, 100, 50), (50, 200, 100), (100, 50, 255), (200, 200, 50),
        (50, 150, 200), (180, 80, 180), (80, 180, 80), (220, 120, 60),
        (60, 120, 220), (150, 150, 150),
    ]

    for i in range(count):
        path = video_dir / f"synthetic_clip_{i:02d}.mp4"
        if path.exists():
            paths.append(path)
            continue

        w, h, fps = 224, 224, 12
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        base = np.array(colors[i % len(colors)], dtype=np.uint8)

        for t in range(fps * 3):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            offset = int(20 * np.sin(t / 3))
            frame[:, :] = base
            frame[max(0, offset):min(h, offset + 80), :] = base * 0.5
            out.write(frame)
        out.release()
        paths.append(path)
        print(f"Generated synthetic video: {path.name}")

    return paths


def download_sample_videos(
    video_dir: str | Path,
    urls: list[str] | None = None,
) -> list[Path]:
    """Download sample MP4 files; fall back to synthetic clips if download fails."""
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    urls = urls or DEFAULT_SAMPLE_URLS
    paths = []
    failed = 0

    for i, url in enumerate(urls):
        filename = url.split("/")[-1]
        dest = video_dir / filename
        if dest.exists() and dest.stat().st_size > 1000:
            paths.append(dest)
            continue
        print(f"Downloading video {i + 1}/{len(urls)}: {filename}")
        if _download_one(url, dest):
            paths.append(dest)
        else:
            failed += 1

    if len(paths) < 3:
        print(f"Only {len(paths)} videos downloaded — generating synthetic fallback clips...")
        paths = generate_synthetic_videos(video_dir, count=10)

    print(f"Ready: {len(paths)} video clips in {video_dir}")
    return paths


def get_clip_path(video_dir: Path, clip_id: int) -> Path | None:
    video_dir = Path(video_dir)
    mp4s = sorted(video_dir.glob("*.mp4"))
    if not mp4s:
        return None
    return mp4s[clip_id % len(mp4s)]
