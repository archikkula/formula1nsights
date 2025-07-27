// frontend/src/pages/Home.js

import React, { useState, useEffect } from "react";

export default function Home() {
  const [news,   setNews]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetch("/api/news")
      .then(r => {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      })
      .then(data => { setNews(data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <p className="text-center">Loading news…</p>;
  if (error)   return <p className="text-center text-red-600">Error: {error}</p>;
  if (!news.length) return <p className="text-center">No news items yet.</p>;

    return (
      
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
      {news.map((item, i) => (
        <div
          key={i}
          className="bg-white rounded-lg overflow-hidden shadow-lg flex flex-col"
        >
          {item.image_url && (
            <img
              src={item.image_url}
              alt={item.headline}
              className="h-48 w-full object-cover"
            />
          )}
          <div className="p-4 flex-1 flex flex-col">
            <h3 className="font-f1-bold text-xl text-f1-black mb-2">
              {item.headline}
            </h3>
            <p className="text-gray-700 text-sm flex-1 mb-4">
              {item.summary}
            </p>
            <a
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-auto text-f1-red hover:underline"
            >
              Read more →
            </a>
          </div>
        </div>
      ))}
    </div>
  );
}
