// src/Watchlist.js
import React from "react";

/*
 Props:
  - watchlist: [{MovieID, title, genres, rating}, ...]
  - removeFromWatchlist(movieID)
*/
export default function Watchlist({ watchlist = [], removeFromWatchlist }) {
  return (
    <div>
      {watchlist.length === 0 && <div className="muted">No movies added yet</div>}

      {watchlist.map((m) => (
        <div className="watch-item" key={m.MovieID}>
          <div>
            <div className="watch-title">{m.title}</div>
            <div className="watch-genres muted">{m.genres}</div>
            <div className="watch-rating">Your rating: {m.rating}</div>
          </div>

          <div>
            <button className="btn-remove" onClick={() => removeFromWatchlist(m.MovieID)}>
              Remove
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
