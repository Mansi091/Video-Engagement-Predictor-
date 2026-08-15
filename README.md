# Video Engagement Prediction Pipeline

This project is a **Video Engagement Predictor** that figures out whether a user will like and engage with a specific video. It acts like a recommendation system, but it's built to test how much better predictions get when you analyze the visual content of the video itself, rather than just relying on user history.

It predicts user engagement by combining:
1. **User Behavior:** Analyzing watch history, past ratings, and synthetic session interaction features (like skip frequency).
2. **Visual Video Analysis:** Using **CLIP video analysis** to "watch" and extract 512-dimensional visual features from the video frames.

Built for the Sony Research India Data Science Intern position (video analysis + predictive ML + scale).

## Datasets (all free, no API keys)

| Dataset | Source | What we use it for |
|---------|--------|-------------------|
| **MovieLens latest-small** | [GroupLens](https://grouplens.org/datasets/movielens/latest/) | Real user ratings → engagement labels |
| **Google sample videos** | [CC sample bucket](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/) | 10 MP4 clips for CLIP feature extraction |
| **Synthetic interactions** | Generated locally | Session events (watch, skip, pause) |

See **[DATASETS.md](DATASETS.md)** for full details, URLs, and join logic.

## Models

| Step | Model |
|------|--------|
| Video features | **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) — free, MIT license |
| Baseline predictor | **XGBoost/GradientBoosting** on user + interaction features only |
| Full predictor | **GradientBoosting** + **PyTorch MLP** with CLIP embeddings |

## Quick Start

```powershell
cd C:\Users\Mansi\OneDrive\Desktop\video-engagement-pipeline
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Quick test (~1 min, skips CLIP and runs on a small data sample)
.\.venv\Scripts\python.exe scripts/run_pipeline.py --quick

# Full run (downloads videos + runs CLIP on the full dataset)
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

## Pipeline

```
MovieLens ratings + movies metadata
        +
Sample MP4 clips (10 videos)
        ↓
CLIP frame encoding → 512-dim video embedding per clip
        ↓
Join with user history + session interaction features
        ↓
Train XGBoost baseline vs user+video models
        ↓
Offline eval (AUC, F1) + bias report by genre segment
```

## Project Structure

```
video-engagement-pipeline/
├── DATASETS.md          # Dataset sources and join logic
├── config.yaml
├── src/video_engagement/
│   ├── data/            # MovieLens download, video download, interactions
│   ├── video/           # CLIP encoder
│   ├── models/          # PyTorch MLP + sklearn baselines
│   └── pipeline/        # End-to-end runner
├── scripts/run_pipeline.py
└── experiments/results/
```

## License

MIT (code). MovieLens and sample videos have their own licenses — see DATASETS.md.
