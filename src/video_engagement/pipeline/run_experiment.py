"""End-to-end video engagement prediction experiment."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_engagement.data.download_movielens import download_movielens, build_engagement_dataset
from video_engagement.data.download_videos import download_sample_videos
from video_engagement.data.generate_interactions import (
    build_user_history_features,
    encode_genres,
    generate_interaction_logs,
)
from video_engagement.models.predictor import (
    bias_report,
    train_mlp,
    train_xgboost_baseline,
)
from video_engagement.video.clip_encoder import CLIPVideoEncoder


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or PROJECT_ROOT / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def run_experiment(config: dict | None = None, skip_clip: bool = False) -> dict:
    config = config or load_config()
    data_cfg = config["data"]
    results_dir = PROJECT_ROOT / config["experiments"]["output_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Video Engagement Prediction Pipeline")
    print("=" * 60)

    # Step 1: Download MovieLens
    print("\n[1/6] Downloading MovieLens latest-small (GroupLens)...")
    ml_dir = download_movielens(
        PROJECT_ROOT / data_cfg["raw_dir"],
        url=data_cfg.get("movielens_url"),
    )
    ratings = build_engagement_dataset(
        ml_dir,
        engagement_threshold=data_cfg["engagement_threshold"],
        max_users=data_cfg.get("max_users"),
        max_ratings=data_cfg.get("max_ratings"),
        seed=data_cfg["seed"],
    )
    print(f"  Ratings loaded: {len(ratings)} rows, {ratings['userId'].nunique()} users")

    # Step 2: Download sample videos
    print("\n[2/6] Downloading sample video clips (Google CC bucket)...")
    video_dir = PROJECT_ROOT / data_cfg["video_dir"]
    download_sample_videos(video_dir, urls=config["videos"]["sample_urls"])

    # Step 3: Interaction logs + user features
    print("\n[3/6] Generating interaction logs and user features...")
    interactions = generate_interaction_logs(ratings, seed=data_cfg["seed"])
    user_feats = build_user_history_features(ratings)
    movies = pd.read_csv(ml_dir / "movies.csv")
    genre_feats = encode_genres(movies)

    df = interactions.merge(user_feats, on="userId", how="left")
    df = df.merge(genre_feats, on="movieId", how="left")

    interaction_cols = ["watch_ratio", "skip_count", "session_events"]
    user_cols = ["user_avg_rating", "user_rating_count", "user_high_engagement_rate"]
    genre_cols = [c for c in df.columns if c.startswith("genre_")]

    # Step 4: CLIP video features
    print("\n[4/6] Extracting CLIP video features (openai/clip-vit-base-patch32)...")
    clip_cfg = config["clip"]
    vid_cfg = config["videos"]

    if skip_clip:
        print("  Skipping CLIP — using random embeddings for quick test")
        clip_embeddings = {i: np.random.randn(512).astype(np.float32) for i in range(10)}
    else:
        encoder = CLIPVideoEncoder(
            model_name=clip_cfg["model_name"],
            batch_size=clip_cfg["batch_size"],
        )
        clip_embeddings = encoder.encode_all_clips(
            video_dir,
            frames_per_video=vid_cfg["frames_per_video"],
            frame_interval_sec=vid_cfg["frame_interval_sec"],
        )

    emb_cols = [f"clip_{i}" for i in range(512)]
    emb_rows = []
    for clip_id, emb in clip_embeddings.items():
        row = {"video_clip_id": clip_id}
        for i, v in enumerate(emb):
            row[f"clip_{i}"] = v
        emb_rows.append(row)
    emb_df = pd.DataFrame(emb_rows)
    df = df.merge(emb_df, on="video_clip_id", how="left")

    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / "training_data.parquet", index=False)

    # Step 5: Train models
    print("\n[5/6] Training engagement predictors...")
    y = df["high_engagement"].values.astype(np.float32)

    X_user_only = df[user_cols + interaction_cols + genre_cols].fillna(0).values
    X_full = df[user_cols + interaction_cols + genre_cols + emb_cols].fillna(0).values

    baseline_result = train_xgboost_baseline(X_user_only, y, seed=data_cfg["seed"])
    full_xgb = train_xgboost_baseline(X_full, y, seed=data_cfg["seed"])
    full_xgb.name = "xgboost_user_plus_video"

    model_cfg = config["model"]
    _, mlp_result = train_mlp(
        X_full, y,
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        epochs=model_cfg["epochs"],
        batch_size=model_cfg["batch_size"],
        lr=model_cfg["learning_rate"],
        val_fraction=model_cfg["val_fraction"],
        seed=data_cfg["seed"],
    )

    # Bias report by genre bucket
    top_genre = df[[c for c in genre_cols if c != "genre_(no genres listed)"]].idxmax(axis=1)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=data_cfg["seed"])
    probs = cross_val_predict(gb, X_full, y, cv=5, method="predict_proba")[:, 1]
    bias = bias_report(y, probs, top_genre.values)

    # Step 6: Results
    print("\n[6/6] Saving results...")
    results = {
        "timestamp": datetime.now().isoformat(),
        "datasets": {
            "engagement": "MovieLens latest-small (GroupLens)",
            "videos": "Google CC sample bucket (10 MP4 clips)",
            "interactions": "Synthetic session logs (local)",
        },
        "n_samples": len(df),
        "engagement_rate": float(y.mean()),
        "models": [
            {"name": baseline_result.name, "auc": baseline_result.auc, "f1": baseline_result.f1, "accuracy": baseline_result.accuracy},
            {"name": full_xgb.name, "auc": full_xgb.auc, "f1": full_xgb.f1, "accuracy": full_xgb.accuracy},
            {"name": mlp_result.name, "auc": mlp_result.auc, "f1": mlp_result.f1, "accuracy": mlp_result.accuracy},
        ],
        "ablation": {
            "user_only_auc": baseline_result.auc,
            "user_plus_video_auc": full_xgb.auc,
            "video_feature_lift": full_xgb.auc - baseline_result.auc,
        },
        "bias_report": bias,
    }

    results_path = results_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [m["name"] for m in results["models"]]
    aucs = [m["auc"] for m in results["models"]]
    ax.barh(names, aucs, color="#3498db", alpha=0.85)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Engagement Prediction — Model Comparison")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Samples: {results['n_samples']}, engagement rate: {results['engagement_rate']:.1%}")
    for m in results["models"]:
        print(f"  {m['name']}: AUC={m['auc']:.3f}, F1={m['f1']:.3f}")
    print(f"  Video feature lift (AUC): {results['ablation']['video_feature_lift']:+.3f}")
    print(f"\nResults saved to {results_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP download for quick test")
    parser.add_argument("--quick", action="store_true", help="Small data + skip CLIP")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg["data"]["max_users"] = 100
        cfg["data"]["max_ratings"] = 500
        cfg["model"]["epochs"] = 5
        args.skip_clip = True
    run_experiment(cfg, skip_clip=args.skip_clip)
