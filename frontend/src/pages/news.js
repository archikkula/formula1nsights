import React, { useState, useEffect } from 'react';

export default function News() {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  useEffect(() => {
    fetch('/api/news')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setNews(data);
        setLastUpdated(new Date());
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const formatLastUpdated = (date) => {
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) return <p className="text-center text-white">Loading news…</p>;
  if (error)   return <p className="text-center text-red-500">Error: {error}</p>;
  if (!news.length) return <p className="text-center text-white">No news items yet.</p>;

  return (
    <div className="px-4 md:px-8 py-8">
    
      <div className="relative mb-12 text-center">
        
        {/* Main header content */}
        <div className="relative z-10">
          <h1 className="font-f1-display text-4xl md:text-5xl lg:text-6xl font-bold text-f1-red mb-4 animate-fade-in-up text-shadow-news">
            F1 News Hub
          </h1>
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 text-f1-white">
            <p className="text-lg font-f1-display animate-fade-in-up-delay">
              Latest Formula 1 Headlines
            </p>
            <div className="flex items-center gap-2 animate-pulse-subtle">
              <div className="w-2 h-2 bg-f1-red rounded-full animate-ping"></div>
              <span className="text-sm text-gray-300">
                Last updated: {formatLastUpdated(lastUpdated)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* News grid */}
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

      {/* CSS Animations */}
      <style jsx>{`

        @keyframes pulse-subtle {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.7;
          }
        }

        .animate-slide-right {
          animation: slide-right 20s linear infinite;
        }

        .animate-fade-in-up {
          animation: fade-in-up 0.8s ease-out;
        }

        .animate-fade-in-up-delay {
          animation: fade-in-up 0.8s ease-out 0.3s both;
        }

        .animate-pulse-subtle {
          animation: pulse-subtle 2s ease-in-out infinite;
        }

        .text-shadow-news {
          text-shadow: 
            0 0 10px rgba(19, 19, 19, 0.8),
            0 0 20px rgba(19, 19, 19, 0.6),
            2px 2px 4px rgba(19, 19, 19, 0.9);
        }
      `}</style>
    </div>
  );
}