import os
import json
import hashlib
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests

RSS_URL = (
    "https://news.google.com/rss/search?q={query}"
    "&hl=en-IN&gl=IN&ceid=IN:en"
)
SEARCHES = [
    "India stock market Nifty Sensex when:1d",
    "NSE BSE Indian stocks earnings results when:1d",
    "RBI India inflation interest rates rupee when:1d",
    "FII DII flows Indian stock market when:1d",
    "US markets Federal Reserve crude oil India when:1d",
    "Indian stocks corporate announcements bulk deals when:1d",
]
MAX_ARTICLES = 25
STATE_FILE = "seen_news.json"
HF_URL = "https://router.huggingface.co/v1/chat/completions"


def load_seen():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data if isinstance(data, list) else [])
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_seen(ids):
    # Keep a bounded history of IDs.
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(ids)[-5000:], f, ensure_ascii=False, indent=2)


def fetch_articles(seen_ids):
    collected = {}
    for query in SEARCHES:
        url = RSS_URL.format(query=quote_plus(query))
        try:
            r = requests.get(
                url, timeout=25,
                headers={"User-Agent": "IndiaMarketNewsMonitor/1.0"}
            )
            r.raise_for_status()
            root = ElementTree.fromstring(r.content)
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                source_node = item.find("source")
                source = (
                    (source_node.text or "").strip()
                    if source_node is not None else "Google News"
                )
                if not title or not link:
                    continue
                article_id = hashlib.sha256(link.encode("utf-8")).hexdigest()
                if article_id in seen_ids:
                    continue
                try:
                    published_dt = parsedate_to_datetime(pub_date)
                    if published_dt.tzinfo is None:
                        published_dt = published_dt.replace(tzinfo=timezone.utc)
                    published_sort = published_dt.timestamp()
                    published_text = published_dt.astimezone().strftime(
                        "%Y-%m-%d %H:%M %Z"
                    )
                except Exception:
                    published_sort = 0
                    published_text = pub_date or "Time unavailable"
                collected[article_id] = {
                    "id": article_id,
                    "title": title,
                    "link": link,
                    "published": published_text,
                    "published_sort": published_sort,
                    "source": source,
                }
        except Exception as exc:
            print(f"RSS feed warning for '{query}': {exc}")

    articles = sorted(
        collected.values(),
        key=lambda a: a["published_sort"],
        reverse=True
    )[:MAX_ARTICLES]
    return articles


def ai_analysis(articles):
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return (
            "AI analysis was skipped because HF_TOKEN is not configured.\n"
            "The headlines below are still included. Check the original "
            "sources before making trading decisions."
        )

    model = (os.getenv("HF_MODEL") or "Qwen/Qwen2.5-7B-Instruct:cheapest").strip()
    headlines = "\n".join(
        f"- {a['title']} | Source: {a['source']} | "
        f"Published: {a['published']} | Link: {a['link']}"
        for a in articles
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You analyze news for Indian equity-market traders. "
                    "Use only the provided headlines and metadata; do not "
                    "pretend you read linked articles or know live prices. "
                    "Summarize the most material stories for Nifty 50, "
                    "Sensex, Bank Nifty, sectors, and individual Indian "
                    "stocks. For each item label possible impact as "
                    "potentially positive, negative, mixed, or unclear; "
                    "explain the reasoning and what needs verification. "
                    "Separate reported facts from inference. Do not give "
                    "personalized buy/sell instructions or invent targets. "
                    "Mention uncertainty and end with an intraday risk "
                    "checklist. If evidence is insufficient, say so."
                )
            },
            {
                "role": "user",
                "content": "Analyze these recent headlines:\n" + headlines
            }
        ],
        "max_tokens": 1100,
        "temperature": 0.2
    }
    try:
        response = requests.post(
            HF_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return (
            f"AI analysis failed for model '{model}': {exc}\n"
            "Headlines are included below; check the source links directly. "
            "Verify model availability and Hugging Face Inference Provider "
            "quota/token permissions if this keeps happening."
        )


def make_report(articles, analysis):
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        f"INDIAN STOCK-MARKET NEWS — {now}",
        "=" * 65,
        "",
        f"New articles found: {len(articles)}",
        "",
    ]
    if not articles:
        lines += [
            "No new articles were found since the previous successful run.",
            "This does not mean there is no market-moving news.",
            "",
        ]
    else:
        for idx, a in enumerate(articles, 1):
            lines += [
                f"{idx}. {a['title']}",
                f"   Source: {a['source']}",
                f"   Published: {a['published']}",
                f"   Link: {a['link']}",
                "",
            ]
    lines += [
        "AI MARKET-IMPACT SUMMARY",
        "-" * 65,
        analysis,
        "",
        "Risk note: This is automated headline analysis, not a trading "
        "signal. Headlines may be incomplete or misleading. Verify source "
        "articles, live prices, liquidity, and risk before trading.",
    ]
    return "\n".join(lines)


def send_email(subject, body):
    host = os.getenv("SMTP_HOST") or "smtp.gmail.com"
    port = int(os.getenv("SMTP_PORT") or "465")
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_APP_PASSWORD", "").strip()
    recipient = os.getenv("EMAIL_TO", "").strip()

    missing = [
        name for name, value in [
            ("SMTP_USER", user),
            ("SMTP_APP_PASSWORD", password),
            ("EMAIL_TO", recipient),
        ] if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing GitHub Actions secret(s): " + ", ".join(missing)
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = recipient
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
        server.login(user, password)
        server.send_message(msg)


def main():
    seen_ids = load_seen()
    articles = fetch_articles(seen_ids)
    analysis = ai_analysis(articles) if articles else (
        "No new headlines to analyze this hour."
    )
    report = make_report(articles, analysis)
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    subject = f"India Market News | {now} | {len(articles)} new"
    print(report)
    send_email(subject, report)

    # Mark only after email delivery succeeds, so a failed email can retry
    # next run without silently losing the articles.
    seen_ids.update(a["id"] for a in articles)
    save_seen(seen_ids)
    print("Email sent successfully.")


if __name__ == "__main__":
    main()
