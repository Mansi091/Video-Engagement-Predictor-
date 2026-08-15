# Datasets — Sources & Usage

This project uses **two free public datasets** plus a small bundled video sample.

---

## Dataset 1: MovieLens Latest Small (engagement labels)

| Field | Detail |
|-------|--------|
| **Source** | GroupLens Research, University of Minnesota |
| **URL** | https://grouplens.org/datasets/movielens/latest/ |
| **Direct download** | https://files.grouplens.org/datasets/movielens/ml-latest-small.zip |
| **License** | Free for research/education (see GroupLens terms) |
| **Size** | ~1 MB zip |
| **Auth required** | No |

**What we use:**
- `ratings.csv` — userId, movieId, rating (0.5–5.0), timestamp
- `movies.csv` — movieId, title, genres
- `tags.csv` — user-applied tags (annotations)

**Engagement label:**
- `high_engagement = 1` if rating >= 4.0, else `0`
- Optional regression target: normalized rating

**Why:** Real human preference signal — proxy for “did the user engage with this content?”

---

## Dataset 2: Google Sample Videos (video files for CLIP)

| Field | Detail |
|-------|--------|
| **Source** | Google Creative Commons sample bucket |
| **Base URL** | http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ |
| **License** | Creative Commons (sample/demo content) |
| **Auth required** | No |

**Clips downloaded (10 short MP4s):**
- BigBuckBunny.mp4, ElephantsDream.mp4, ForBiggerBlazes.mp4, etc.

**Fallback:** If downloads fail (network/403), the pipeline auto-generates 10 short synthetic MP4 clips locally so CLIP extraction always runs.

---

## Dataset 3: Synthetic user interaction logs (video sessions)

| Field | Detail |
|-------|--------|
| **Source** | Generated locally by `generate_interactions.py` |
| **Based on** | MovieLens user IDs + mapped video clips |

**What we generate:**
- Session events: watch, skip, pause, replay, exit
- watch_ratio, skip_count per user-video pair
- Joined with MovieLens ratings where movieId maps to a sample video

**Why:** Mimics high-dimensional interaction data (skip patterns, watch depth) that MovieLens alone doesn't provide.

---

## How datasets are joined

```
MovieLens ratings (userId, movieId, rating)
        +
movies.csv (genres, title)
        +
Sample video clip (movieId % 10 → one of 10 MP4s)
        ↓
CLIP extracts visual features from MP4
        ↓
User history features from past ratings
        ↓
Train predictor: engagement ~ video_features + user_features
```

**Note:** MovieId → video mapping is a **demo proxy** (each movie maps to a sample clip by `movieId % 10`). For production, you'd use actual per-title trailer URLs from TMDB/IMDB. This is documented in the README.

---

## Optional: UCF101 (full video analysis scale-up)

| Field | Detail |
|-------|--------|
| **Source** | University of Central Florida |
| **Access** | `torchvision.datasets.UCF101(download=True)` |
| **Size** | ~6.5 GB |
| **License** | Academic/research use |

Enable in `config.yaml` with `use_ucf101: true` for larger-scale CLIP extraction (not required for quick run).
