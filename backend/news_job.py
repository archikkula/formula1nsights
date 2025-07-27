import os
import feedparser
from datetime import datetime
from models import NewsItem
from twilio.rest import Client
import smtplib

TWILIO_SID   = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM  = os.getenv("TWILIO_FROM")
TWILIO_TO    = os.getenv("TWILIO_TO")
EMAIL_USER   = os.getenv("NEWS_EMAIL_USER")
EMAIL_PASS   = os.getenv("NEWS_EMAIL_PASS")
SMTP_SERVER  = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", 587))

def fetch_headlines():
    url = "https://www.formula1.com/en/latest/all.xml"
    feed = feedparser.parse(url)

    items = []
    for e in feed.entries:
        title   = e.title
        link    = e.link
        summary = getattr(e, "summary", "")[:200] 
        img = None
        if "media_content" in e:
            img = e.media_content[0]["url"]
        elif "media_thumbnail" in e:
            img = e.media_thumbnail[0]["url"]
        items.append((title, link, summary, img))
    return items

# def send_sms(msg):
#     Client(TWILIO_SID, TWILIO_TOKEN).messages.create(
#         body=msg,
#         from_=TWILIO_FROM,
#         to=TWILIO_TO
#     )

# def send_email(subject, body):
#     with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
#         smtp.starttls()
#         smtp.login(EMAIL_USER, EMAIL_PASS)
#         msg = f"Subject: {subject}\n\n{body}"
#         smtp.sendmail(EMAIL_USER, EMAIL_USER, msg)

def job():
    from app import Session
    session = Session()

    for title, link, summary, image_url in fetch_headlines():
        exists = session.query(NewsItem).filter_by(url=link).first()
        if not exists:
            ni = NewsItem(
                headline  = title,
                url       = link,
                summary   = summary,
                image_url = image_url
            )
            session.add(ni)
            session.commit()
    session.close()

if __name__ == "__main__":
    job()
