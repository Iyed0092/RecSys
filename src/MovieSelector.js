// src/MovieSelector.js
import React, { useMemo, useState } from "react";

/*
 Props:
  - movies: [{MovieID, title, genres}, ...]
  - onAdd(movie, rating) => function to call when user clicks "Add to watchlist"
*/
export default function MovieSelector({ movies = [], onAdd }) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [rating, setRating] = useState(4);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return movies;
    return movies.filter(
      (m) =>
        (m.title || "").toLowerCase().includes(q) ||
        (m.genres || "").toLowerCase().includes(q)
    );
  }, [movies, query]);

  function handleAdd() {
    if (!selected) return;
    if (typeof onAdd === "function") onAdd(selected, rating);
    // clear selection after add
    setSelected(null);
    setRating(4);
    setQuery("");
  }

  return (
    <div>
      <input
        className="input"
        placeholder="Search title or genre (e.g. Comedy, 1995)..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <div className="movie-scroll">
        {filtered.map((m) => (
          <div
            key={m.MovieID}
            className={`movie-row ${selected && selected.MovieID === m.MovieID ? "active" : ""}`}
            onClick={() => setSelected(m)}
            title={m.title}
          >
            <div className="movie-row-left">
              <div className="thumb">{m.MovieID}</div>
              <div className="meta">
                <div className="title">{m.title}</div>
                <div className="genres">{m.genres}</div>
              </div>
            </div>

            <div className="movie-row-actions">
              <button
                className="btn-ghost"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelected(m);
                }}
              >
                View
              </button>
            </div>
          </div>
        ))}
      </div>

      {selected && (
        <div className="selected-card">
          <div className="selected-title">{selected.title}</div>
          <div className="selected-genres">Genres: {selected.genres}</div>

          <div className="rating-row">
            <label>Rating:</label>
            <div className="stars">
              {[1, 2, 3, 4, 5].map((s) => (
                <span
                  key={s}
                  className={`star ${s <= rating ? "on" : ""}`}
                  onClick={() => setRating(s)}
                >
                  ★
                </span>
              ))}
            </div>
          </div>

          <div className="selected-actions">
            <button className="btn-primary" onClick={handleAdd}>
              Add to watchlist
            </button>
            <button
              className="btn-clear"
              onClick={() => {
                setSelected(null);
                setRating(4);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
