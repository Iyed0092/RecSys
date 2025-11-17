# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import time
from threading import Lock
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)  # allow requests from frontend (e.g. localhost:3000)

# ---------------------------
# Configuration - update paths if needed
# ---------------------------
START_NEW_USER_ID = 6041
MOVIES_FILE = r"C:\Users\iyedm\recommender\src\movies.dat"
RATINGS_FILE = r"C:\Users\iyedm\recommender\src\ratings.dat"

# ---------------------------
# Global state
# ---------------------------
_lock = Lock()
NEW_USER_ID = None  # will be initialized lazily on first add

# ---------------------------
# Load data (MovieLens .dat with '::' separator)
# ---------------------------
if not os.path.exists(MOVIES_FILE):
    raise FileNotFoundError(f"movies file not found: {MOVIES_FILE}")
if not os.path.exists(RATINGS_FILE):
    raise FileNotFoundError(f"ratings file not found: {RATINGS_FILE}")

# movies.dat format: MovieID::Title::Genres
movies = pd.read_csv(
    MOVIES_FILE,
    delimiter="::",
    header=None,
    names=["MovieID", "Title", "Genres"],
    engine="python",
    encoding="latin-1",
)
movies["MovieID"] = movies["MovieID"].astype(int)

# ratings.dat format: UserID::MovieID::Rating::Timestamp
ratings = pd.read_csv(
    RATINGS_FILE,
    delimiter="::",
    header=None,
    names=["UserID", "MovieID", "Rating", "Timestamp"],
    engine="python",
    encoding="latin-1",
)
ratings["UserID"] = ratings["UserID"].astype(int)
ratings["MovieID"] = ratings["MovieID"].astype(int)
ratings["Rating"] = ratings["Rating"].astype(float)
# Ensure Timestamp column exists and is int
if "Timestamp" not in ratings.columns:
    ratings["Timestamp"] = int(time.time())
else:
    # convert to int if possible
    try:
        ratings["Timestamp"] = ratings["Timestamp"].astype(int)
    except Exception:
        ratings["Timestamp"] = int(time.time())

# ---------------------------
# Item-item recommender (based on your function)
# ---------------------------
def item_item_based_recommendation(user_id, movies_per_movie=5, k_users=500, k_movies=None):
    """
    Returns a DataFrame with columns: ['MovieID', 'PredictedRating', 'title', 'genres']
    Sorted by PredictedRating descending.
    """
    global ratings, movies

    if k_movies is None:
        k_movies = movies.shape[0]

    # reduce problem size: select top-k users and top-k movies by frequency
    top_k_users = ratings["UserID"].value_counts().nlargest(k_users).index
    top_k_movies = ratings["MovieID"].value_counts().nlargest(k_movies).index
    ratings_topk = ratings[ratings["UserID"].isin(top_k_users) & ratings["MovieID"].isin(top_k_movies)]

    # pivot: rows = MovieID, cols = UserID
    pivot_table = ratings_topk.pivot(index="MovieID", columns="UserID", values="Rating").fillna(0)

    if pivot_table.shape[0] == 0:
        # no data to compute similarity
        return pd.DataFrame(columns=["MovieID", "PredictedRating", "title", "genres"])

    # compute item-item similarity (movies x movies)
    X = csr_matrix(pivot_table.values)
    item_sim = cosine_similarity(X)  # shape (n_movies, n_movies)

    index_to_movie = list(pivot_table.index)  # movie ids in pivot order
    movie_to_index = {mid: idx for idx, mid in enumerate(index_to_movie)}

    # get watched movies for this user from the global ratings dataframe
    watched_rows = ratings[ratings["UserID"] == user_id]
    watched_movies_tuples = [(int(r.MovieID), float(r.Rating)) for r in watched_rows.itertuples()]

    if len(watched_movies_tuples) == 0:
        # user hasn't rated anything (shouldn't happen for new_user after adding) -> return empty
        return pd.DataFrame(columns=["MovieID", "PredictedRating", "title", "genres"])

    # gather candidate similar movies
    similar_movies = set()
    for watched_mid, _ in watched_movies_tuples:
        if watched_mid not in movie_to_index:
            continue
        idx = movie_to_index[watched_mid]
        sorted_idxs = np.argsort(-item_sim[idx])  # descending by similarity
        # take top neighbors (skip itself)
        neighbor_idxs = sorted_idxs[1 : movies_per_movie + 1]
        for ni in neighbor_idxs:
            similar_mid = index_to_movie[ni]
            similar_movies.add(similar_mid)

    watched_movie_ids = [m for (m, _) in watched_movies_tuples]
    candidate_unwatched = [mid for mid in similar_movies if mid not in watched_movie_ids]

    # prediction: weighted average of ratings on watched movies using item similarity
    def predict_movie_rating(movie_id):
        if movie_id not in movie_to_index:
            return 0.0
        m_idx = movie_to_index[movie_id]
        numerator = 0.0
        denom = 0.0
        for (w_mid, w_rating) in watched_movies_tuples:
            if w_mid not in movie_to_index:
                continue
            w_idx = movie_to_index[w_mid]
            sim = float(item_sim[m_idx, w_idx])
            numerator += sim * w_rating
            denom += abs(sim)
        return (numerator / denom) if denom != 0.0 else 0.0

    predicted = []
    for mid in candidate_unwatched:
        pred = predict_movie_rating(mid)
        predicted.append((int(mid), float(pred)))

    # sort and create dataframe
    predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
    preds_df = pd.DataFrame(predicted, columns=["MovieID", "PredictedRating"])

    # merge movie info; ensure lowercase keys for frontend compatibility
    preds_df = preds_df.merge(movies, on="MovieID", how="left")
    preds_df = preds_df.rename(columns={"Title": "title", "Genres": "genres"})
    # keep only desired columns
    preds_df = preds_df[["MovieID", "PredictedRating", "title", "genres"]]
    return preds_df

# ---------------------------
# API endpoints
# ---------------------------
@app.route("/api/movies", methods=["GET"])
def api_movies():
    """
    Returns the movie list. Each item contains MovieID, Title, Genres.
    Frontend maps Title/Genres -> title/genres as needed.
    """
    # Return a few columns for the frontend
    out = movies[["MovieID", "Title", "Genres"]].to_dict(orient="records")
    return jsonify(out)


@app.route("/api/add_and_recommend", methods=["POST"])
def api_add_and_recommend():
    """
    Accepts:
      - { "movie_id": <int>, "rating": <float> }
      - OR { "watched": [ { "movie_id": <int>, "rating": <float> }, ... ] }

    Appends rows to ratings DataFrame for a persistent NEW_USER_ID (starting at START_NEW_USER_ID),
    runs item_item_based_recommendation for that new user, and returns:
      { new_user_id, added_rows: [...], recommendations: [...] }
    """
    global NEW_USER_ID, ratings

    payload = request.get_json(force=True)
    if payload is None:
        return jsonify({"error": "invalid JSON payload"}), 400

    entries = []
    # support both single and batch payloads
    if "watched" in payload and isinstance(payload["watched"], list):
        for w in payload["watched"]:
            if ("movie_id" in w or "MovieID" in w) and ("rating" in w or "Rating" in w):
                movie_id = int(w.get("movie_id") or w.get("MovieID"))
                rating_val = float(w.get("rating") or w.get("Rating"))
                entries.append({"MovieID": movie_id, "Rating": rating_val})
    elif "movie_id" in payload and "rating" in payload:
        entries.append({"MovieID": int(payload["movie_id"]), "Rating": float(payload["rating"])})
    else:
        return jsonify({"error": "invalid payload shape"}), 400

    if len(entries) == 0:
        return jsonify({"error": "no valid entries found"}), 400

    with _lock:
        # set NEW_USER_ID on first use
        if NEW_USER_ID is None:
            highest = int(ratings["UserID"].max())
            NEW_USER_ID = max(START_NEW_USER_ID, highest + 1)

        ts = int(time.time())
        new_rows = []
        for e in entries:
            new_row = {
                "UserID": int(NEW_USER_ID),
                "MovieID": int(e["MovieID"]),
                "Rating": float(e["Rating"]),
                "Timestamp": int(ts),
            }
            new_rows.append(new_row)

        # append to global ratings DataFrame (in-memory)
        ratings = pd.concat([ratings, pd.DataFrame(new_rows)], ignore_index=True)

    # run recommender
    try:
        rec_df = item_item_based_recommendation(NEW_USER_ID)
    except Exception as ex:
        return jsonify({"error": "recommendation failed", "detail": str(ex)}), 500

    # prepare output
    added_rows_out = [
        {"UserID": r["UserID"], "MovieID": r["MovieID"], "Rating": r["Rating"], "Timestamp": r["Timestamp"]}
        for r in new_rows
    ]
    # rec_df columns: MovieID, PredictedRating, title, genres
    # convert numpy types to native python types
    recommendations_out = []
    for row in rec_df.to_dict(orient="records"):
        recommendations_out.append(
            {
                "MovieID": int(row.get("MovieID", 0)),
                "title": row.get("title") if row.get("title") is not None else "",
                "genres": row.get("genres") if row.get("genres") is not None else "",
                "PredictedRating": float(row.get("PredictedRating", 0.0)),
            }
        )

    return jsonify({"new_user_id": int(NEW_USER_ID), "added_rows": added_rows_out, "recommendations": recommendations_out})


if __name__ == "__main__":
    print("Starting Flask app. Listening on http://0.0.0.0:5000")
    print(f"Movies: {MOVIES_FILE}\nRatings: {RATINGS_FILE}")
    app.run(host="0.0.0.0", port=5000, debug=True)
