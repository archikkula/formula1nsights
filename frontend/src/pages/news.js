import React, { useState, useEffect } from 'react';

export default function News() {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/news')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setNews(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <p className="text-center text-white">Loading news…</p>;
  if (error)   return <p className="text-center text-red-500">Error: {error}</p>;
  if (!news.length) return <p className="text-center text-white">No news items yet.</p>;

  return (
    <div className="px-4 md:px-8 py-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {news.map((item, i) => (
          <div
            key={i}
            className="bg-f1-black p-6 rounded-lg"
          >
            {item.image_url && (
              <img
                src={item.image_url}
                alt={item.headline}
                className="h-48 w-full object-cover rounded-md mb-4"
              />
            )}
            <h3 className="text-2xl font-bold text-white mb-2">
              {item.headline}
            </h3>
            <p className="text-sm font-light text-white mb-4">
              {item.summary}
            </p>
            <a
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-f1-red hover:underline font-medium"
            >
              Read more →
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}
