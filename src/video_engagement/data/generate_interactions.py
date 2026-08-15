"""Generate synthetic session-level interaction logs."""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_TYPES = ["watch", "skip", "pause", "replay", "exit"]


def generate_interaction_logs(
    ratings_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate per user-video session interaction features.

    Correlates watch behavior with engagement label so features are informative
    but noisy (realistic).
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _, row in ratings_df.iterrows():
        engaged = row["high_engagement"]
        n_events = rng.integers(5, 15)
        watch_count = 0
        skip_count = 0

        for _ in range(n_events):
            if engaged:
                event = rng.choice(EVENT_TYPES, p=[0.5, 0.1, 0.1, 0.15, 0.15])
            else:
                event = rng.choice(EVENT_TYPES, p=[0.2, 0.35, 0.1, 0.05, 0.30])
            if event == "watch":
                watch_count += 1
            if event == "skip":
                skip_count += 1

        total = watch_count + skip_count + 1
        watch_ratio = watch_count / total

        rows.append({
            "userId": row["userId"],
            "movieId": row["movieId"],
            "video_clip_id": row["video_clip_id"],
            "watch_ratio": watch_ratio,
            "skip_count": skip_count,
            "session_events": n_events,
            "high_engagement": engaged,
            "rating": row["rating"],
        })

    return pd.DataFrame(rows)


def build_user_history_features(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user historical engagement stats."""
    agg = (
        ratings_df.groupby("userId")
        .agg(
            user_avg_rating=("rating", "mean"),
            user_rating_count=("rating", "count"),
            user_high_engagement_rate=("high_engagement", "mean"),
        )
        .reset_index()
    )
    return agg


def encode_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode MovieLens genre strings."""
    all_genres = set()
    for genres in movies_df["genres"].dropna():
        all_genres.update(genres.split("|"))

    genre_df = movies_df[["movieId", "genres"]].copy()
    for g in sorted(all_genres):
        genre_df[f"genre_{g}"] = genre_df["genres"].apply(
            lambda x: int(g in str(x).split("|")) if pd.notna(x) else 0
        )
    return genre_df.drop(columns=["genres"])
