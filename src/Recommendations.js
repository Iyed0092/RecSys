// src/Recommendations.js
import React from "react";

/*
 Props:
  - recommendations: [{MovieID, title, genres, PredictedRating}, ...]
*/
export default function Recommendations({ recommendations = [] }) {
  if (!recommendations || recommendations.length === 0) {
    return <div className="muted">No recommendations yet</div>;
  }

  return (
    <div>
      <div style={{ marginBottom: 8 }} className="muted">Top recommendations (Predicted Rating)</div>
      {recommendations.map((r) => (
        <div className="reco-item" key={r.MovieID}>
          <div style={{ maxWidth: "78%" }}>
            <div className="reco-title">{r.title}</div>
            <div className="reco-genres muted">{r.genres}</div>
          </div>
          <div className="reco-score">
            {(typeof r.PredictedRating === "number") ? r.PredictedRating.toFixed(2) : (r.PredictedRating ?? "-")}
          </div>
        </div>
      ))}
    </div>
  );
}
