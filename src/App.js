import React, { useEffect, useState } from "react";
import MovieSelector from "./MovieSelector";
import Watchlist from "./Watchlist";
import Recommendations from "./Recommendations";
import "./App.css";

// Update this to your Flask backend URL
const BACKEND_URL = "http://localhost:5000";

export default function App() {
  const [movies, setMovies] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loadingMovies, setLoadingMovies] = useState(true);
  const [error, setError] = useState(null);
  const [newUserId, setNewUserId] = useState(null);

  // Load movies from backend
  useEffect(() => {
    setLoadingMovies(true);
    fetch(`${BACKEND_URL}/api/movies`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch movies (${res.status})`);
        return res.json();
      })
      .then((data) => {
        // MovieLens columns: MovieID, Title, Genres → map to lowercase for frontend
        const mapped = data.map((m) => ({
          MovieID: m.MovieID,
          title: m.Title || m.title,
          genres: m.Genres || m.genres,
        }));
        setMovies(mapped);
      })
      .catch((err) => setError(err.message || "Failed to load movies"))
      .finally(() => setLoadingMovies(false));
  }, []);

  // Add a movie with rating
  async function addMovieAndGetRec(movie, rating) {
    if (!movie || !rating) return;
    try {
      // Optimistic update local watchlist
      setWatchlist((prev) => {
        const exists = prev.find((p) => p.MovieID === movie.MovieID);
        if (exists) {
          return prev.map((p) =>
            p.MovieID === movie.MovieID ? { ...p, rating } : p
          );
        }
        return [...prev, { ...movie, rating }];
      });

      // POST to backend
      const res = await fetch(`${BACKEND_URL}/api/add_and_recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ movie_id: movie.MovieID, rating }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server returned ${res.status}: ${text}`);
      }

      const payload = await res.json();
      // Response: { new_user_id, added_rows, recommendations }
      if (payload.new_user_id) setNewUserId(payload.new_user_id);
      if (payload.recommendations) setRecommendations(payload.recommendations);
    } catch (err) {
      console.error("Add + recommend failed:", err);
      setError(err.message || "Failed to add movie");
    }
  }

  function removeFromLocalWatchlist(movieID) {
    setWatchlist((prev) => prev.filter((m) => m.MovieID !== movieID));
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>MovieLens — select watched movies & rate them</h1>
        <div className="muted small">
          Backend endpoints: GET /api/movies | POST /api/add_and_recommend
        </div>
        {newUserId && (
          <div className="muted small">Active user id: {newUserId}</div>
        )}
      </header>

      <main className="app-grid">
        <section className="panel panel-left">
          <h2>All movies</h2>
          {loadingMovies && <div className="muted">Loading movies...</div>}
          {error && <div className="error">{error}</div>}
          {!loadingMovies && !error && (
            <MovieSelector
              movies={movies}
              onAdd={(movie, rating) => addMovieAndGetRec(movie, rating)}
            />
          )}
        </section>

        <section className="panel panel-mid">
          <h2>Your current watchlist</h2>
          <Watchlist
            watchlist={watchlist}
            removeFromWatchlist={removeFromLocalWatchlist}
          />
        </section>

        <section className="panel panel-right">
          <h2>Recommendations</h2>
          <Recommendations recommendations={recommendations} />
        </section>
      </main>

      <footer className="app-footer muted" style={{ marginTop: 12 }}>
        The backend appends each rating to the ratings DataFrame and returns recommendations.
      </footer>
    </div>
  );
}
