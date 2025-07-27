from flask import Flask, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, NewsItem
from apscheduler.schedulers.background import BackgroundScheduler
from news_job import job as news_job

app = Flask(__name__)
CORS(app)

engine = create_engine("sqlite:///news.db", echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

@app.route("/api/health")
def health_check():
    return jsonify({"status": "ok"})

@app.route("/health")
def health_check_simple():
    return jsonify({"status": "ok"})

@app.route("/api/news")
def get_news():
    session = Session()
    items = (
      session
      .query(NewsItem)
      .order_by(NewsItem.fetched_at.desc())
      .all()
    )
    session.close()
    return jsonify([
      {
        "headline":  i.headline,
        "url":       i.url,
        "summary":   i.summary,
        "image_url": i.image_url
      }
      for i in items
    ])

@app.route("/news")
def get_news_simple():
    session = Session()
    items = (
      session
      .query(NewsItem)
      .order_by(NewsItem.fetched_at.desc())
      .all()
    )
    session.close()
    return jsonify([
      {
        "headline":  i.headline,
        "url":       i.url,
        "summary":   i.summary,
        "image_url": i.image_url
      }
      for i in items
    ])

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=news_job, trigger="interval", minutes=5)
    scheduler.start()
    try:
        app.run(debug=True, port=5001)
    finally:
        scheduler.shutdown()
