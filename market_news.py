import os
import json
import hashlib
import smtplib
import ssl
import html
from email.message import EmailMessage
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import requests

IST = ZoneInfo("Asia/Kolkata")
RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
SEARCHES = [
    'Nifty 50 Indian stock market when:1d',
    'Bank Nifty private banks banking results RBI when:1d',
    'India crude oil energy power oil gas stocks when:1d',
    'India commodities metals gold copper aluminium stocks when:1d',
    'NSE BSE Indian stocks earnings results corporate announcements when:1d',
    'India inflation rupee FII DII government policy global markets when:1d',
]
MAX_ARTICLES = 35
MAX_AI_ARTICLES = 22
STATE_FILE = "seen_news.json"
HF_URL = "https://router.huggingface.co/v1/chat/completions"

# Only these publishers are treated as the primary approved sources.
TIER1 = {
    "nse", "nse india", "national stock exchange", "bse", "bse india",
    "bombay stock exchange", "sebi", "reserve bank of india", "rbi",
    "press information bureau", "pib", "ministry of statistics", "mospi",
    "ministry of finance", "government of india",
}
TIER2 = {"reuters", "reuters india", "cnbc-tv18", "cnbc tv18", "economic times", "the economic times", "et markets"}


def now_ist():
    return datetime.now(IST)


def normalise_source(source):
    return " ".join((source or "").lower().replace("®", "").split()).strip(" .,-")


def classify_source(source):
    s = normalise_source(source)
    if any(x == s or x in s for x in TIER1):
        return "Tier 1 — Official / Primary"
    if any(x == s or x in s for x in TIER2):
        return "Tier 2 — Established Reporting"
    return "Needs Validation — Other Source"


def load_seen():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data if isinstance(data, list) else [])
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_seen(ids):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(ids)[-5000:], f, ensure_ascii=False, indent=2)


def fetch_articles(seen_ids):
    collected = {}
    for query in SEARCHES:
        url = RSS_URL.format(query=quote_plus(query))
        try:
            response = requests.get(url, timeout=25, headers={"User-Agent": "IndiaMarketNewsMonitor/2.0"})
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                source_node = item.find("source")
                source = ((source_node.text or "").strip() if source_node is not None else "Unknown source")
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
                    published_text = published_dt.astimezone(IST).strftime("%d %b %Y, %I:%M %p IST")
                except Exception:
                    published_sort, published_text = 0, pub_date or "Time unavailable"
                collected[article_id] = {
                    "id": article_id, "title": title, "link": link,
                    "published": published_text, "published_sort": published_sort,
                    "source": source, "tier": classify_source(source),
                }
        except Exception as exc:
            print(f"RSS feed warning for '{query}': {exc}")

    # Priority order: approved source tiers first, then other sources for validation.
    priority = {"Tier 1 — Official / Primary": 0, "Tier 2 — Established Reporting": 1, "Needs Validation — Other Source": 2}
    articles = sorted(collected.values(), key=lambda a: (priority[a["tier"]], -a["published_sort"]))
    return articles[:MAX_ARTICLES]


def ai_analysis(articles):
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return "AI analysis skipped: HF_TOKEN is not configured. Headlines and links are included below; verify source material and price action manually."
    model = (os.getenv("HF_MODEL") or "Qwen/Qwen2.5-7B-Instruct:cheapest").strip()
    selected = articles[:MAX_AI_ARTICLES]
    headlines = "\n".join(
        f"{i}. [{a['tier']}] {a['title']} | Publisher: {a['source']} | Published: {a['published']} | Link: {a['link']}"
        for i, a in enumerate(selected, 1)
    )
    prompt = (
        "Create a concise hourly Indian cash-equity trader briefing for a beginner. "
        "Use ONLY the supplied headlines and metadata. Do not claim to have read articles, "
        "verified filings, or seen live prices. For each story refer to its exact headline or number. "
        "Output sections in this order: (1) Executive Market Snapshot, (2) Your List — Nifty 50 and Bank Nifty, "
        "(3) Your List — Energy and Commodity-related stocks, (4) Your List — Other market-relevant news, "
        "(5) Recommended Checks — other stocks/sectors worth investigating, (6) Needs Validation — non-approved sources, "
        "(7) Key Notes and Intraday Risk Checklist. For each material story give: a 1-3 sentence headline-based summary; "
        "possible trading impact and reasoning; possible index/sector relevance; Bias (Bullish/Bearish/Neutral/Mixed/Unclear); "
        "Impact (High/Medium/Low); stocks to investigate with rationale only if reasonably supported by the headline; "
        "and what to verify in the original report, exchange filing, price, volume and chart. Never invent ticker symbols, "
        "targets, support/resistance, current prices, or facts not in headlines. Clearly label inference as possible, "
        "not confirmed. An impact rating is importance, not probability of price direction. If a headline does not support "
        "a stock-specific conclusion, say 'No stock-specific conclusion from headline alone'. Separate other publishers under "
        "Needs Validation. Prefer Tier 1 and Tier 2. Keep concise and do not provide buy/sell instructions.\n\nHEADLINES:\n"
        + headlines
    )
    payload = {"model": model, "messages": [
        {"role": "system", "content": "You are a cautious Indian equity-market news analyst. Accuracy and explicit uncertainty are more important than filling every field."},
        {"role": "user", "content": prompt}], "max_tokens": 2600, "temperature": 0.15}
    try:
        response = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return (f"AI analysis failed for model '{model}': {exc}.\n"
                "Headline links are still included. Verify source articles and check HF_TOKEN, model access and quota.")


def grouped_articles(articles):
    return {
        "Tier 1 — Official / Primary": [a for a in articles if a["tier"] == "Tier 1 — Official / Primary"],
        "Tier 2 — Established Reporting": [a for a in articles if a["tier"] == "Tier 2 — Established Reporting"],
        "Needs Validation — Other Source": [a for a in articles if a["tier"] == "Needs Validation — Other Source"],
    }


def article_html(a):
    title = html.escape(a["title"])
    link = html.escape(a["link"], quote=True)
    return (f'<li style="margin:0 0 12px 0"><a href="{link}"><b>{title}</b></a><br>'
            f'<small>Source: {html.escape(a["source"])} | Published: {html.escape(a["published"])}</small><br>'
            f'<small>Source class: {html.escape(a["tier"])}</small></li>')


def make_report(articles, analysis):
    now = now_ist()
    groups = grouped_articles(articles)
    lines = [
        f"INDIA MARKET INTELLIGENCE | {now.strftime('%d %b %Y')} | {now.strftime('%I:%M %p IST')} (UTC+05:30)",
        "News cutoff: report generation time shown above. Headlines are not a substitute for reading the original source.",
        f"New headlines: {len(articles)} | Tier 1: {len(groups['Tier 1 — Official / Primary'])} | Tier 2: {len(groups['Tier 2 — Established Reporting'])} | Needs validation: {len(groups['Needs Validation — Other Source'])}",
        "",
        "SOURCE LINKS — APPROVED SOURCES FIRST",
    ]
    for group_name in ("Tier 1 — Official / Primary", "Tier 2 — Established Reporting", "Needs Validation — Other Source"):
        lines += ["", group_name, "-" * 55]
        if not groups[group_name]:
            lines.append("No new headlines in this source category.")
        for a in groups[group_name]:
            lines += [f"• {a['title']}", f"  Publisher: {a['source']} | Published: {a['published']}", f"  Link: {a['link']}"]
        if group_name == "Needs Validation — Other Source":
            lines.append("Validation required: confirm through an exchange filing, company/regulator release or established reporting before relying on these items.")
    lines += ["", "TRADING-FOCUSED ANALYSIS", "-" * 55, analysis,
              "", "PRIORITY ORDER", "1. Nifty 50 and Bank Nifty; 2. Energy and commodity-linked stocks; 3. Other sectors with a verified catalyst or notable trend.",
              "", "RISK AND SOURCE NOTES",
              "Tier 1 sources: NSE, BSE, SEBI, RBI, PIB, MOSPI and other clearly identified official sources. Tier 2 sources: Reuters, CNBC-TV18 and Economic Times Markets.",
              "The source label is based on the publisher name supplied by Google News RSS; confirm the actual publisher and source page. Exchange-hosted company disclosures are company-submitted and may still require interpretation or confirmation.",
              "Bias and impact are automated headline-based assessments, not price forecasts or trading signals. Check the linked report, live price, volume, liquidity, chart context and risk before any trade."]
    plain = "\n".join(lines)

    # HTML version adds clickable headlines while retaining the same content in plain-text fallback.
    parts = [
        '<html><body style="font-family:Arial,sans-serif;line-height:1.45;color:#202124">',
        f'<h2>India Market Intelligence | {now.strftime("%d %b %Y")} | {now.strftime("%I:%M %p IST")} (UTC+05:30)</h2>',
        f'<p><b>Report generated:</b> {now.strftime("%d %B %Y, %I:%M %p IST")}<br><b>News cutoff:</b> report generation time.</p>',
        f'<p><b>New headlines:</b> {len(articles)} | <b>Tier 1:</b> {len(groups["Tier 1 — Official / Primary"])} | <b>Tier 2:</b> {len(groups["Tier 2 — Established Reporting"])} | <b>Needs validation:</b> {len(groups["Needs Validation — Other Source"])}</p>',
        '<h3>1. Executive Market Snapshot</h3><p>See the trading-focused analysis below for headline-based market context. Verify current index levels and live price action separately.</p>',
        '<h3>2. Your List — Nifty 50 and Bank Nifty / 3. Energy and Commodities / 4. Other Market-Relevant News</h3>',
        '<p>AI analysis prioritises these groups in the order above. Direct headline links are grouped by source quality below.</p>',
        '<h3>Source Links — Tier 1: Official / Primary</h3><ul>' + ''.join(article_html(a) for a in groups['Tier 1 — Official / Primary']) + '</ul>',
        '<h3>Source Links — Tier 2: Established Reporting</h3><ul>' + ''.join(article_html(a) for a in groups['Tier 2 — Established Reporting']) + '</ul>',
        '<h3>Needs Validation — Other Sources</h3><p>These items are not in the approved Tier 1/2 publisher list. Validate independently before relying on them.</p><ul>' + ''.join(article_html(a) for a in groups['Needs Validation — Other Source']) + '</ul>',
        '<h3>Trading-Focused Analysis</h3><pre style="white-space:pre-wrap;font-family:Arial,sans-serif">' + html.escape(analysis) + '</pre>',
        '<h3>Recommended Checks</h3><p>Investigate only names supported by a credible catalyst. Confirm exchange/company filings, price reaction, relative volume, liquidity and broader index/sector direction. No automatic buy/sell signal is implied.</p>',
        '<h3>Key Notes and Intraday Risk Checklist</h3><ul><li>Check whether the news is already priced in.</li><li>Verify the original report and timestamp.</li><li>Confirm price and volume rather than trading on a headline alone.</li><li>Define a stop and maximum acceptable loss before entering a trade.</li><li>For unapproved publishers, validate independently first.</li></ul>',
        '<p><small>Bias/impact labels are automated analysis, not forecasts. NSE/BSE company disclosures may be company-submitted; read the filing itself. This email is informational, not personalised investment advice.</small></p>',
        '</body></html>'
    ]
    return plain, ''.join(parts)


def send_email(subject, plain_body, html_body):
    host = os.getenv("SMTP_HOST") or "smtp.gmail.com"
    port = int(os.getenv("SMTP_PORT") or "465")
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_APP_PASSWORD", "").strip()
    recipient = os.getenv("EMAIL_TO", "").strip()
    missing = [name for name, value in [("SMTP_USER", user), ("SMTP_APP_PASSWORD", password), ("EMAIL_TO", recipient)] if not value]
    if missing:
        raise RuntimeError("Missing GitHub Actions secret(s): " + ", ".join(missing))
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = recipient
    msg.set_content(plain_body)
    msg.add_alternative(html_body, subtype="html")
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
        server.login(user, password)
        server.send_message(msg)


def main():
    seen_ids = load_seen()
    articles = fetch_articles(seen_ids)
    analysis = ai_analysis(articles) if articles else (
        "No new headlines found since the previous successful run. This does not mean there is no market-moving news. "
        "Review official exchange filings and current market data if making decisions."
    )
    plain, html_body = make_report(articles, analysis)
    now = now_ist()
    subject = f"India Market Intelligence | {now.strftime('%d %b %Y')} | {now.strftime('%I:%M %p IST')} | {len(articles)} new"
    print(plain)
    send_email(subject, plain, html_body)
    # Save only after successful email delivery, preserving retry behaviour.
    seen_ids.update(a["id"] for a in articles)
    save_seen(seen_ids)
    print("Email sent successfully.")


if __name__ == "__main__":
    main()
