from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
import datetime

Base = declarative_base()

class NewsItem(Base):
    __tablename__ = "news"

    id         = Column(Integer, primary_key=True)
    headline   = Column(String, unique=True, nullable=False)
    url        = Column(String, unique=True, nullable=False)
    summary    = Column(String, nullable=True)
    image_url  = Column(String, nullable=True)
    fetched_at = Column(DateTime, default=datetime.datetime.utcnow)
