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
CORS(app)  


START_NEW_USER_ID = 6041
MOVIES_FILE = r"C:\Users\iyedm\recommender\src\movies.dat"
RATINGS_FILE = r"C:\Users\iyedm\recommender\src\ratings.dat"


_lock = Lock()
NEW_USER_ID = None  


if not os.path.exists(MOVIES_FILE):
    raise FileNotFoundError(f"movies file not found: {MOVIES_FILE}")
if not os.path.exists(RATINGS_FILE):
    raise FileNotFoundError(f"ratings file not found: {RATINGS_FILE}")

movies = pd.read_csv(
    MOVIES_FILE,
    delimiter="::",
    header=None,
    names=["MovieID", "Title", "Genres"],
    engine="python",
    encoding="latin-1",
)
movies["MovieID"] = movies["MovieID"].astype(int)

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
if "Timestamp" not in ratings.columns:
    ratings["Timestamp"] = int(time.time())
else:
    try:
        ratings["Timestamp"] = ratings["Timestamp"].astype(int)
    except Exception:
        ratings["Timestamp"] = int(time.time())


def get_key_from_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None

movie_indexer = {k:v for k,v in enumerate(movies["MovieID"])}
movies["Genres"] = movies["Genres"].str.replace("|", " ")

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
content_based_genre_matrix = tfidf.fit_transform(movies['Genres'])


def user_profile(user_id):
    user_movies = ratings[ratings['UserID'] == user_id]
    movies_ids = list(user_movies['MovieID'].astype(int).values)
    movies_idx = [get_key_from_value(movie_indexer, id) for id in movies_ids]
    ratings_values = user_movies['Rating'].values
    ratings_vecs = ratings_values[:, np.newaxis]
    movies_vecs = content_based_genre_matrix[movies_idx ,:]

    profile = movies_vecs.multiply(ratings_vecs).mean(axis=0)
    profile_array = np.array(profile)
    return profile_array



def content_based(user_id, top_n=10):
    similarities = []
    for i in range(movies.shape[0]):
        user_prof = user_profile(user_id)
        movie_vector = content_based_genre_matrix[i,:].toarray()
        sim = cosine_similarity(user_prof, movie_vector)
        similarities.append((i, sim))
    sorted_sim= sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    recommended_movie_indices = [idx for idx, sim in sorted_sim]
    recommend_movies_ids = [movie_indexer[index] for index in recommended_movie_indices]
    return movies[movies["MovieID"].isin(recommend_movies_ids)]



def hybrid_based_recommendation(user_id, movies_per_movie=5, k_users=500, k_movies=None):

    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix
    from sklearn.preprocessing import normalize

    global ratings, movies

    if k_movies is None:
        k_movies = movies['MovieID'].nunique()

    top_k_users = ratings["UserID"].value_counts().nlargest(k_users).index
    top_k_movies = ratings["MovieID"].value_counts().nlargest(k_movies).index
    ratings_topk = ratings[ratings["UserID"].isin(top_k_users) & ratings["MovieID"].isin(top_k_movies)].copy()

    if ratings_topk.empty:
        return pd.DataFrame(columns=["MovieID", "PredictedRating", "title", "genres"])

    pivot_table = ratings_topk.pivot(index="MovieID", columns="UserID", values="Rating").fillna(0)

    try:
        movies_idx = movies.set_index("MovieID").loc[pivot_table.index]
    except KeyError:
        movies_idx = movies[movies["MovieID"].isin(pivot_table.index)].set_index("MovieID").loc[pivot_table.index]

    genres_series = movies_idx.get('Genres') if 'Genres' in movies_idx.columns else movies_idx.get('genres')
    genres_series = genres_series.fillna('')

    tfidf = TfidfVectorizer(min_df=1) 
    content_based_genre_matrix = tfidf.fit_transform(genres_series.values)  

    X_ratings = csr_matrix(pivot_table.values)        
    X_hybrid = hstack([X_ratings, content_based_genre_matrix], format='csr') 

    X_norm = normalize(X_hybrid, norm='l2', axis=1)   
    item_sim = X_norm.dot(X_norm.T)                   

    index_to_movie = list(pivot_table.index)        
    movie_to_index = {mid: idx for idx, mid in enumerate(index_to_movie)}

    watched_rows = ratings[ratings["UserID"] == user_id]
    watched_movies_tuples = [(int(r.MovieID), float(r.Rating)) for r in watched_rows.itertuples()]

    if len(watched_movies_tuples) == 0:
        return pd.DataFrame(columns=["MovieID", "PredictedRating", "title", "genres"])

    similar_movies = set()
    for watched_mid, _ in watched_movies_tuples:
        if watched_mid not in movie_to_index:
            continue
        idx = movie_to_index[watched_mid]
        sim_row = item_sim.getrow(idx).toarray().ravel()
        sorted_idxs = np.argsort(-sim_row)
        neighbor_idxs = [i for i in sorted_idxs if i != idx][:movies_per_movie]
        for ni in neighbor_idxs:
            similar_mid = index_to_movie[ni]
            similar_movies.add(similar_mid)

    watched_movie_ids = [m for (m, _) in watched_movies_tuples]
    candidate_unwatched = [mid for mid in similar_movies if mid not in watched_movie_ids]
    if not candidate_unwatched:
        return pd.DataFrame(columns=["MovieID", "PredictedRating", "title", "genres"])

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
            sim_val = item_sim[m_idx, w_idx]
            try:
                sim = float(sim_val)
            except Exception:
                sim = float(sim_val.toarray().item())
            numerator += sim * w_rating
            denom += abs(sim)
        return (numerator / denom) if denom != 0.0 else 0.0

    predicted = []
    for mid in candidate_unwatched:
        pred = predict_movie_rating(mid)
        predicted.append((int(mid), float(pred)))

    predicted = sorted(predicted, key=lambda x: x[1], reverse=True)
    preds_df = pd.DataFrame(predicted, columns=["MovieID", "PredictedRating"])

    title_col = "Title" if "Title" in movies.columns else ("title" if "title" in movies.columns else None)
    genres_col = "Genres" if "Genres" in movies.columns else ("genres" if "genres" in movies.columns else None)
    merge_cols = []
    if title_col:
        merge_cols.append(title_col)
    if genres_col:
        merge_cols.append(genres_col)

    if merge_cols:
        preds_df = preds_df.merge(movies[["MovieID"] + merge_cols], on="MovieID", how="left")

    rename_map = {}
    if title_col:
        rename_map[title_col] = "title"
    if genres_col:
        rename_map[genres_col] = "genres"
    preds_df = preds_df.rename(columns=rename_map)

    keep_cols = ["MovieID", "PredictedRating"]
    if "title" in preds_df.columns:
        keep_cols.append("title")
    if "genres" in preds_df.columns:
        keep_cols.append("genres")

    preds_df = preds_df[keep_cols]
    user_user_recs = user_user_based_recommendation(user_id)
    df_combined = pd.concat([preds_df, user_user_recs], axis=0, ignore_index=True)

    return df_combined



import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import normalize

def user_user_based_recommendation(user_id, top_n=50, k_users=200, k_movies=200, n_neighbors=50):

    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    global ratings, movies

    top_k_users = list(ratings['UserID'].value_counts().nlargest(k_users).index)
    if user_id not in top_k_users:
        top_k_users = top_k_users + [user_id]  

    top_k_movies = set(ratings['MovieID'].value_counts().nlargest(k_movies).index)
    user_movie_ids = set(ratings[ratings['UserID'] == user_id]['MovieID'].unique())
    selected_movies = list(top_k_movies.union(user_movie_ids))

    ratings_topk = ratings[ratings['UserID'].isin(top_k_users) & ratings['MovieID'].isin(selected_movies)].copy()
    if ratings_topk.empty:
        return pd.DataFrame(columns=["MovieID", "PredictedRating", "title", "genres"])

    pivot_users = ratings_topk.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
    user_ids = list(pivot_users.index)
    user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}

    if user_id not in user_to_index:
        return pd.DataFrame(columns=["MovieID", "PredictedRating", "title", "genres"])

    X = csr_matrix(pivot_users.values)  
    Xn = normalize(X, axis=1)           
    user_sim = Xn.dot(Xn.T)             

    u_idx = user_to_index[user_id]
    sim_row = user_sim.getrow(u_idx).toarray().ravel()  # dense 1D array

    sorted_idxs = np.argsort(-sim_row)
    neighbor_idxs = [i for i in sorted_idxs if i != u_idx][:n_neighbors]
    neighbor_user_ids = [user_ids[i] for i in neighbor_idxs]
    neighbor_sim_map = {user_ids[i]: float(sim_row[i]) for i in neighbor_idxs}

    watched_by_target = set(ratings_topk[ratings_topk['UserID'] == user_id]['MovieID'].unique())
    neighbors_ratings = ratings_topk[ratings_topk['UserID'].isin(neighbor_user_ids)].copy()
    neighbors_ratings = neighbors_ratings[~neighbors_ratings['MovieID'].isin(watched_by_target)]

    if neighbors_ratings.empty:
        return pd.DataFrame(columns=["MovieID", "PredictedRating", "title", "genres"])

    global_means = ratings_topk.groupby('MovieID')['Rating'].mean().to_dict()
    global_mean_overall = ratings_topk['Rating'].mean() if len(ratings_topk) else 3.5

    movie_scores = {}
    movie_weights = {}
    for row in neighbors_ratings.itertuples(index=False):
        uid = row.UserID
        mid = row.MovieID
        r = float(row.Rating)
        sim = neighbor_sim_map.get(uid, 0.0)
        if sim <= 0:
            continue
        movie_scores[mid] = movie_scores.get(mid, 0.0) + sim * r
        movie_weights[mid] = movie_weights.get(mid, 0.0) + sim

    preds = []
    for mid, num in movie_scores.items():
        denom = movie_weights.get(mid, 0.0)
        if denom > 0:
            score = num / denom
        else:
            score = float(global_means.get(mid, global_mean_overall))
        preds.append((int(mid), float(score)))

    if not preds:
        popular = (ratings_topk[~ratings_topk['MovieID'].isin(watched_by_target)]
                   .groupby('MovieID')['Rating'].mean()
                   .sort_values(ascending=False)
                   .head(top_n))
        preds = [(int(mid), float(val)) for mid, val in popular.items()]

    preds_df = pd.DataFrame(preds, columns=["MovieID", "PredictedRating"])
    preds_df = preds_df.sort_values("PredictedRating", ascending=False).head(top_n)
    preds_df = preds_df.merge(movies, on="MovieID", how="left")


    rename_map = {}
    if "Title" in preds_df.columns:
        rename_map["Title"] = "title"
    if "Genres" in preds_df.columns:
        rename_map["Genres"] = "genres"
    preds_df = preds_df.rename(columns=rename_map)

    keep_cols = ["MovieID", "PredictedRating"]
    if "title" in preds_df.columns:
        keep_cols.append("title")
    if "genres" in preds_df.columns:
        keep_cols.append("genres")

    return preds_df[keep_cols].reset_index(drop=True)





@app.route("/api/movies", methods=["GET"])
def api_movies():

    out = movies[["MovieID", "Title", "Genres"]].to_dict(orient="records")
    return jsonify(out)


@app.route("/api/add_and_recommend", methods=["POST"])
def api_add_and_recommend():

    global NEW_USER_ID, ratings

    payload = request.get_json(force=True)
    if payload is None:
        return jsonify({"error": "invalid JSON payload"}), 400

    entries = []
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

        ratings = pd.concat([ratings, pd.DataFrame(new_rows)], ignore_index=True)

    try:
        rec_df = hybrid_based_recommendation(NEW_USER_ID)
    except Exception as ex:
        return jsonify({"error": "recommendation failed", "detail": str(ex)}), 500

    added_rows_out = [
        {"UserID": r["UserID"], "MovieID": r["MovieID"], "Rating": r["Rating"], "Timestamp": r["Timestamp"]}
        for r in new_rows
    ]

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
