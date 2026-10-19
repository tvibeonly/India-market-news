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
    # India: indices, policy, filings, sectors and company-specific catalysts
    'Nifty 50 Bank Nifty Nifty Financial Services India markets when:1d',
    'site:nseindia.com corporate announcements results India listed companies when:2d',
    'site:bseindia.com corporate announcements listed companies when:2d',
    'site:rbi.org.in policy monetary policy liquidity India when:7d',
    'site:sebi.gov.in circular order regulation markets when:7d',
    'site:pib.gov.in economy policy energy trade India when:2d',
    'India banking financial services earnings credit deposits RBI when:1d',
    'India crude oil natural gas refinery oil marketing energy power stocks when:1d',
    'India metals mining steel copper aluminium commodity stocks when:1d',
    'India IT technology auto pharma FMCG infrastructure defence real estate stocks when:1d',
    'India inflation GDP PMI rupee bond yields FII DII budget trade policy when:1d',
    # Global catalysts that can transmit into Indian markets
    'Federal Reserve US inflation Treasury yields dollar global markets when:1d',
    'China economy stimulus manufacturing metals commodities markets when:1d',
    'Europe ECB Bank of England Japan Bank of Japan central bank markets when:1d',
    'Middle East geopolitics oil shipping sanctions crude prices when:1d',
    'OPEC oil supply EIA inventories energy markets when:1d',
    'global equities bond yields currencies banking stress trade tariffs sanctions when:1d',
    'Reuters global markets India stocks economy when:1d',
]
MAX_ARTICLES = 60
MAX_AI_ARTICLES = 35
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
    model = (os.getenv("HF_MODEL") or "openai/gpt-oss-120b:cheapest").strip()
    selected = articles[:MAX_AI_ARTICLES]
    headlines = "\n".join(
        f"{i}. [PUBLISHER SOURCE CLASS: {a['tier']}] {a['title']} | Publisher: {a['source']} | Published: {a['published']} | Link: {a['link']}"
        for i, a in enumerate(selected, 1)
    )
    prompt = (
        "Create an hourly global-and-Indian market-intelligence report for a beginner intraday equity trader. "
        "CRITICAL OUTPUT RULE: use the exact template below for EVERY supplied headline, individually. Do not only "
        "write a general sector summary or combine unrelated headlines. Keep each news item separate and numbered. "
        "If a field cannot be supported by the headline/metadata, write 'Unavailable from supplied headline' or "
        "'Needs verification'; do not guess. You have not read full articles and have not checked live market data. "
        "Keep every supplied headline's publisher, published timestamp and direct link attached to its item. "
        "The published timestamp is RSS metadata and may be inaccurate. Distinguish event time from publication time. "
        "Publisher source class is not the same as event-impact tier. Determine event tier separately; source labels "
        "are provisional based on Google News RSS metadata.\n\n"
        "REPORT HEADER: Executive summary (top 3 developments, market-wide risk, scheduled catalysts only if known).\n\n"
        "FOR EVERY NEWS ITEM, USE THIS EXACT STRUCTURE:\n"
        "NEWS IMPACT REPORT — Item [number]\n"
        "Headline: [headline]\n"
        "Source and timing: Publisher; publication time in IST; event time if explicitly known (otherwise 'Unavailable'); "
        "direct source link.\n"
        "News classification: Tier 1 / Tier 2 / Validation required, with one-sentence reason. Tier 1 means critical "
        "potentially immediate market impact; Tier 2 means important development; use Validation required for "
        "unverified, conflicting, weakly sourced, speculative or insufficiently supported claims.\n"
        "Validation: Officially confirmed / Credibly corroborated / Unverified / Cannot determine from supplied metadata. "
        "Do not claim official confirmation unless supplied evidence supports it.\n"
        "Global market impact: Affected markets; affected sectors; relevant global stocks and tickers only if a credible "
        "exposure exists; expected bias (Bullish/Bearish/Mixed/Uncertain) and reasoning. If no clear connection, say so.\n"
        "Indian market impact: Affected indices (Nifty 50, Bank Nifty, Nifty Financial Services or others only if relevant); "
        "affected sectors; relevant NSE/BSE stocks and tickers only where credible exposure exists; expected bias "
        "(Bullish/Bearish/Mixed/Uncertain) and reasoning. If no clear direct Indian connection, say so.\n"
        "Trading interpretation: Potential impact; likely horizon (intraday/short-term/medium-term/unclear); evidence that "
        "would confirm the view (e.g. actual price/volume, index or sector relative strength, USD/INR, yields or commodities "
        "when relevant); and what could invalidate/reverse it. Do not invent observed market data.\n"
        "Action: Monitor / Wait for confirmation / Avoid acting until verified, with a brief reason. Never issue an "
        "automatic buy/sell call, guaranteed direction, or price target. News impact is not a trade signal.\n"
        "Confidence: Low/Medium/High, with reason; default to Low when only a headline is available.\n\n"
        "After all individual items, add: (1) Cross-item sector themes; (2) upcoming catalysts only when supported by "
        "the supplied data, otherwise say 'Not established from supplied headlines'; (3) data-quality/system status and "
        "any limitations. Prioritise Nifty 50 and Bank Nifty, then energy/oil/gas and commodities, then other sectors. "
        "Map stocks based on business exposure, not popularity. Candidate examples are not mandatory lists. Do not force "
        "stock mapping. Never fabricate current prices, volume, FII/DII flows, event times, source URLs, or corroboration. "
        "For every uncertain story, clearly mark uncertainty.\n\nHEADLINES:\n" + headlines
    )
    payload = {"model": model, "messages": [
        {"role": "system", "content": "You are a cautious Indian equity-market news analyst. Accuracy and explicit uncertainty are more important than filling every field."},
        {"role": "user", "content": prompt}], "max_tokens": 7000, "temperature": 0.15}
    try:
        response = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        try:
            detail = (exc.response.text or "").strip()[:900] if exc.response is not None else ""
        except Exception:
            detail = ""
        return (f"AI analysis failed for model '{model}' (HTTP {status}).\n"
                f"Provider response: {detail or 'No response detail supplied.'}\n"
                "Headlines and links are still included. Check HF_TOKEN permissions, model/provider availability, "
                "Inference Providers credits/quota, and HF_MODEL. This error does not by itself prove the token is invalid.")
    except Exception as exc:
        return (f"AI analysis failed for model '{model}': {type(exc).__name__}: {exc}.\n"
                "Headlines and links are still included. Check HF_TOKEN, model/provider availability and quota.")


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


def next_scheduled_ist(now):
    # Existing workflow runs at minute 17 UTC = minute 47 IST each hour.
    candidate = now.replace(minute=47, second=0, microsecond=0)
    if candidate <= now:
        from datetime import timedelta
        candidate += timedelta(hours=1)
    return candidate


def make_report(articles, analysis):
    now = now_ist()
    next_run = next_scheduled_ist(now)
    groups = grouped_articles(articles)
    lines = [
        f"INDIA MARKET INTELLIGENCE | {now.strftime('%d %b %Y')} | {now.strftime('%I:%M %p IST')} (UTC+05:30)",
        f"News cutoff / generated: {now.strftime('%d %b %Y, %I:%M %p IST')} | Next scheduled run (approx.): {next_run.strftime('%d %b %Y, %I:%M %p IST')}.",
        "The workflow may be delayed by GitHub Actions. Headlines are not a substitute for reading the original source.",
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
        f'<p><b>Report generated / news cutoff:</b> {now.strftime("%d %B %Y, %I:%M %p IST")}<br><b>Next scheduled run (approx.):</b> {next_run.strftime("%d %B %Y, %I:%M %p IST")}. GitHub Actions can be delayed.</p>',
        f'<p><b>New headlines:</b> {len(articles)} | <b>Tier 1:</b> {len(groups["Tier 1 — Official / Primary"])} | <b>Tier 2:</b> {len(groups["Tier 2 — Established Reporting"])} | <b>Needs validation:</b> {len(groups["Needs Validation — Other Source"])}</p>',
        '<h3>1. Executive Market Snapshot</h3><p>See the trading-focused analysis below for headline-based market context. Verify current index levels and live price action separately.</p>',
        '<h3>2. Your List — Nifty 50 and Bank Nifty / 3. Energy and Commodities / 4. Other Market-Relevant News</h3>',
        '<p>AI analysis prioritises these groups in the order above. Direct headline links are grouped by source quality below.</p>',
        '<h3>Source Links — Approved Official / Primary Publishers (source class, not event impact tier)</h3><ul>' + ''.join(article_html(a) for a in groups['Tier 1 — Official / Primary']) + '</ul>',
        '<h3>Source Links — Established Reporting Publishers (source class, not event impact tier)</h3><ul>' + ''.join(article_html(a) for a in groups['Tier 2 — Established Reporting']) + '</ul>',
        '<h3>Needs Validation — Other Publishers</h3><p>These items are outside the approved publisher list. This source label is separate from the event-impact Tier; validate independently before relying on them.</p><ul>' + ''.join(article_html(a) for a in groups['Needs Validation — Other Source']) + '</ul>',
        '<h3>Cross-item Themes, Validation, Upcoming Catalysts and Data Quality</h3>',
        '<h3>Full Analysis</h3><pre style="white-space:pre-wrap;font-family:Arial,sans-serif">' + html.escape(analysis) + '</pre>',
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
    subject = f"India Market Intelligence | {now.strftime('%d %b %Y')} | {now.strftime('%I:%M %p IST')}"
    print(plain)
    send_email(subject, plain, html_body)
    # Save only after successful email delivery, preserving retry behaviour.
    seen_ids.update(a["id"] for a in articles)
    save_seen(seen_ids)
    print("Email sent successfully.")


if __name__ == "__main__":
    main()
