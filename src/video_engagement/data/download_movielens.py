"""Download MovieLens latest-small from GroupLens."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens(raw_dir: str | Path, url: str = MOVIELENS_URL) -> Path:
    """Download and extract MovieLens latest-small dataset."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-latest-small.zip"
    extract_dir = raw_dir / "movielens"

    if not (extract_dir / "ml-latest-small" / "ratings.csv").exists():
        print(f"Downloading MovieLens from {url}...")
        urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("MovieLens downloaded and extracted.")
    else:
        print("MovieLens already present, skipping download.")

    return extract_dir / "ml-latest-small"


def load_movielens_tables(ml_dir: Path) -> dict[str, pd.DataFrame]:
    ml_dir = Path(ml_dir)
    tables = {}
    for name in ["ratings", "movies", "tags", "links"]:
        path = ml_dir / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def build_engagement_dataset(
    ml_dir: Path,
    engagement_threshold: float = 4.0,
    max_users: int | None = None,
    max_ratings: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build engagement labels from MovieLens ratings.

    high_engagement = 1 if rating >= threshold (default 4.0 stars).
    """
    tables = load_movielens_tables(ml_dir)
    ratings = tables["ratings"].copy()
    movies = tables["movies"].copy()

    if max_users:
        user_ids = ratings["userId"].unique()
        rng = __import__("numpy").random.default_rng(seed)
        keep_users = rng.choice(user_ids, size=min(max_users, len(user_ids)), replace=False)
        ratings = ratings[ratings["userId"].isin(keep_users)]

    if max_ratings and len(ratings) > max_ratings:
        ratings = ratings.sample(n=max_ratings, random_state=seed)

    ratings = ratings.merge(movies, on="movieId", how="left")
    ratings["high_engagement"] = (ratings["rating"] >= engagement_threshold).astype(int)
    ratings["engagement_score"] = ratings["rating"] / 5.0

    # Map each movie to a sample video clip index (0-9)
    ratings["video_clip_id"] = ratings["movieId"] % 10

    return ratings
