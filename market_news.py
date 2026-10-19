import os
import re
import json
import hashlib
import smtplib
import ssl
import html
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher

import requests

# Existing workflow retained: Google News RSS -> AI analysis -> email -> seen-state.
# Configure secrets through environment variables / GitHub Actions Secrets.
IST = ZoneInfo("Asia/Kolkata")
RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

SEARCHES = [
    # India: markets, official policy, filings and sector/company catalysts
    'Nifty 50 Bank Nifty Nifty Financial Services India markets when:1d',
    'site:nseindia.com corporate announcements results India listed companies when:2d',
    'site:bseindia.com corporate announcements listed companies when:2d',
    'site:rbi.org.in policy monetary policy liquidity India when:7d',
    'site:sebi.gov.in circular order regulation markets when:7d',
    'site:pib.gov.in economy policy energy trade India when:2d',
    'site:mospi.gov.in CPI GDP industrial production India when:7d',
    'site:finmin.gov.in OR site:dea.gov.in India budget borrowing taxation markets when:7d',
    'India banking financial services earnings credit deposits RBI when:1d',
    'India crude oil natural gas refinery oil marketing energy power stocks when:1d',
    'India metals mining steel copper aluminium commodity stocks when:1d',
    'India IT technology auto pharma FMCG infrastructure defence real estate stocks when:1d',
    'India inflation GDP PMI rupee bond yields FII DII budget trade policy when:1d',
    'India company earnings order win merger acquisition regulatory action stocks when:1d',
    # Global macro, central banks and official statistics
    'Federal Reserve FOMC US inflation employment Treasury yields dollar markets when:1d',
    'site:bls.gov CPI employment site:bea.gov GDP PCE United States when:7d',
    'ECB Bank of England Bank of Japan China PBOC central bank decisions when:1d',
    'China economy stimulus manufacturing property trade tariffs markets when:1d',
    'IMF World Bank global growth financial stability sovereign debt when:7d',
    # Global geopolitics, trade and energy
    'Middle East geopolitics oil shipping sanctions crude prices when:1d',
    'Russia Ukraine sanctions war trade global markets when:1d',
    'global tariffs trade restrictions export controls supply chain markets when:1d',
    'OPEC oil supply EIA inventories IEA energy markets when:1d',
    'global banking stress credit markets volatility currencies when:1d',
    # Established business and financial reporting
    'Reuters global markets India stocks economy when:1d',
    'Bloomberg global markets central banks commodities when:1d',
    'Financial Times geopolitics trade economy markets when:1d',
    'CNBC world markets global economy when:1d',
    'CNBC-TV18 Indian markets stocks when:1d',
    'Business Standard India economy stocks markets when:1d',
    'Economic Times Markets India stocks when:1d',
]

MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "80"))
MAX_AI_ARTICLES = int(os.getenv("MAX_AI_ARTICLES", "50"))
STATE_FILE = os.getenv("STATE_FILE", "seen_news.json")
HF_URL = "https://router.huggingface.co/v1/chat/completions"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "25"))

TIER1 = {
    "nse", "nse india", "national stock exchange", "bse", "bse india",
    "bombay stock exchange", "sebi", "reserve bank of india", "rbi",
    "press information bureau", "pib", "ministry of statistics", "mospi",
    "ministry of finance", "government of india", "federal reserve",
    "u.s. federal reserve", "european central bank", "ecb", "bank of japan",
    "bank of england", "us bureau of labor statistics", "bureau of economic analysis",
    "u.s. treasury", "eia", "u.s. energy information administration", "iea",
    "international energy agency", "opec", "imf", "international monetary fund",
    "world bank", "eurostat", "european commission", "ustr",
}
TIER2 = {
    "reuters", "reuters india", "cnbc-tv18", "cnbc tv18", "economic times",
    "the economic times", "et markets", "bloomberg", "financial times",
    "cnbc", "associated press", "business standard",
}


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
    # Preserve a bounded state file so it does not grow forever.
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(ids)[-5000:], f, ensure_ascii=False, indent=2)


def parse_published(pub_date):
    try:
        dt = parsedate_to_datetime(pub_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt, dt.timestamp(), dt.astimezone(IST).strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return None, 0, pub_date or "Time unavailable"


def fetch_articles(seen_ids):
    collected = {}
    for query in SEARCHES:
        url = RSS_URL.format(query=quote_plus(query))
        try:
            response = requests.get(
                url, timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "IndiaGlobalMarketNewsMonitor/3.0"}
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                source_node = item.find("source")
                source = ((source_node.text or "").strip()
                          if source_node is not None else "Unknown source")
                if not title or not link:
                    continue

                article_id = hashlib.sha256(link.encode("utf-8")).hexdigest()
                if article_id in seen_ids:
                    continue

                _, published_sort, published_text = parse_published(pub_date)
                collected[article_id] = {
                    "id": article_id,
                    "title": title,
                    "link": link,
                    "published": published_text,
                    "published_raw": pub_date,
                    "published_sort": published_sort,
                    "source": source,
                    "source_class": classify_source(source),
                    "query_context": query,
                }
        except Exception as exc:
            print(f"RSS feed warning for '{query}': {type(exc).__name__}: {exc}")

    priority = {
        "Tier 1 — Official / Primary": 0,
        "Tier 2 — Established Reporting": 1,
        "Needs Validation — Other Source": 2,
    }
    articles = sorted(
        collected.values(),
        key=lambda a: (priority[a["source_class"]], -a["published_sort"])
    )
    return articles[:MAX_ARTICLES]


def build_analysis_prompt(articles):
    selected = articles[:MAX_AI_ARTICLES]
    headlines = "\n".join(
        f"{i}. Title: {a['title']} | Publisher: {a['source']} | "
        f"Publisher class: {a['source_class']} | Published (RSS metadata, IST): {a['published']} | "
        f"URL: {a['link']} | Search context: {a['query_context']}"
        for i, a in enumerate(selected, 1)
    )

    return (
        "Prepare a global-and-Indian market-intelligence report for an intraday equity trader. "
        "You only have the supplied headlines and metadata; do not claim to have read full articles "
        "or checked live market prices unless the input explicitly supplies that evidence.\n\n"

        "CRITICAL CONSOLIDATION RULES:\n"
        "1. CONSOLIDATE related articles about the same underlying event into ONE news event/report. "
        "This includes multiple publishers reporting the same event, multiple headlines about the same "
        "company announcement, and repeated coverage of one policy or geopolitical development.\n"
        "2. Do NOT merge distinct events merely because they affect the same sector or stock. "
        "Separate unrelated catalysts.\n"
        "3. In a consolidated item, use one concise canonical headline and summary, then list ALL relevant "
        "supporting article links, publisher names, and publication timestamps supplied for that event. "
        "Identify the best primary/official source when one is present. Do not invent links.\n"
        "4. If the same stock appears in several different events, keep the events separate, but add a "
        "cross-event stock/sector summary after the individual reports. Never make a duplicate item solely "
        "because a second outlet rewrote the same story.\n"
        "5. If articles conflict, say what conflicts and mark validation as required. Do not silently reconcile.\n"
        "6. Headlines are not full articles. If the facts are insufficient, state 'Needs verification from "
        "the linked article or primary source'. Never claim the event is officially confirmed based only on "
        "a publisher name or RSS metadata.\n\n"

        "SOURCE VS EVENT CLASSIFICATION:\n"
        "Publisher source class (official / established reporting / other) is separate from event impact tier. "
        "For every event, classify impact as Tier 1 (potentially immediate/critical), Tier 2 (important), "
        "or Validation required (unverified/conflicting/insufficiently supported). Explain why.\n\n"

        "MANDATORY REPORT HEADER:\n"
        "1. Executive summary: top 3–5 consolidated developments, overall risk tone, and most important "
        "India/global cross-market transmission channels. Do not infer live market direction without data.\n"
        "2. News counts: number of supplied articles, number of consolidated events, and validation-required events.\n\n"

        "MANDATORY TEMPLATE FOR EACH CONSOLIDATED EVENT:\n"
        "NEWS IMPACT REPORT — Event [number]\n"
        "Canonical headline:\n"
        "Concise summary: What happened, what is known, and what remains unknown.\n"
        "Source and timing: Original/primary source if available; all relevant supporting publishers and direct URLs; "
        "publication timestamps in IST; event time only if explicitly supplied/known. Distinguish event time from "
        "publication time. State when timestamp is unavailable.\n"
        "News classification: Tier 1 / Tier 2 / Validation required, with reason.\n"
        "Validation: Officially confirmed / Credibly corroborated / Unverified / Cannot determine from supplied metadata. "
        "Explain evidence level and any source disagreement.\n"
        "GLOBAL MARKET IMPACT: Countries/regions; affected indices/asset classes; affected sectors; candidate "
        "individual listed stocks with company name, ticker, exchange/country, exposure and potential bias. "
        "Include only stocks with a credible event-specific link. Explain the transmission mechanism. If none, say so.\n"
        "INDIAN MARKET IMPACT: Nifty 50, Bank Nifty, Nifty Financial Services or other indices only if relevant; "
        "affected sectors; individual NSE/BSE stocks with company name, ticker, sector, exposure, potential bias, "
        "and reason. Distinguish direct from indirect effects. If no clear connection, state that explicitly.\n"
        "STOCK-BY-STOCK IMPACT: For each relevant stock, use: Company (ticker, exchange) — potential bias "
        "(bullish/bearish/mixed/uncertain); direct/indirect exposure; why affected; counterargument or key risk. "
        "Avoid forcing a list of popular tickers. Verify symbols only when supported; if uncertain, mark for verification.\n"
        "SECTOR SUMMARY: Most exposed sectors; likely channel (demand, prices, input costs, FX, interest rates, "
        "regulation, supply chain, earnings or sentiment); winners/losers may differ within a sector.\n"
        "TRADING INTERPRETATION: Potential impact horizon (intraday/short-term/medium-term/unclear); evidence needed "
        "to confirm (price/volume/relative strength, yields, FX, commodities, etc. only if available); invalidation "
        "conditions; whether impact may already be priced in.\n"
        "ACTION: Monitor / Wait for confirmation / Avoid acting until verified. No automatic buy/sell call, guaranteed "
        "direction, entry, stop-loss or price target without adequate live data and explicit methodology.\n"
        "CONFIDENCE: Low/Medium/High with reason. When only headlines/metadata are supplied, default to Low or "
        "explain why a stronger confidence is justified by supplied evidence.\n\n"

        "AFTER INDIVIDUAL EVENTS, INCLUDE:\n"
        "A. CONSOLIDATED STOCK WATCHLIST — Indian stocks and global stocks, each listed once, showing the event IDs "
        "that affect it, combined directional assessment (bullish/bearish/mixed/uncertain), key catalysts and risks. "
        "Do not collapse different event causes into one simplistic conclusion.\n"
        "B. CROSS-EVENT SECTOR IMPACT — common themes and conflicting forces across events.\n"
        "C. UPCOMING CATALYSTS — only when explicitly supported by supplied material; otherwise say not established.\n"
        "D. VALIDATION QUEUE — events requiring primary-source confirmation or more article detail.\n"
        "E. DATA QUALITY / SYSTEM LIMITATIONS — headlines not full articles, missing live data, unavailable timestamps, "
        "and other gaps.\n\n"

        "MARKET-MAPPING GUIDANCE (candidates only; never attach all of these automatically):\n"
        "India banks: HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK. India IT: TCS, INFY, HCLTECH, WIPRO. "
        "India energy: RELIANCE, ONGC, OIL, IOC, BPCL, HPCL, IGL, GAIL. Metals: TATASTEEL, JSWSTEEL, HINDALCO, VEDL. "
        "Auto: MARUTI, TATAMOTORS, M&M. Pharma: SUNPHARMA, DRREDDY, CIPLA. Utilities: NTPC, POWERGRID, TATAPOWER. "
        "Infrastructure/capital goods: LT, ABB, SIEMENS. Verify ticker and exposure; select only relevant names.\n"
        "Global candidates when relevant: banks JPM, BAC, C, GS; technology MSFT, NVDA, AAPL, AMZN; energy XOM, CVX, SHEL; "
        "materials BHP, RIO, FCX. Verify the listing/exchange and company exposure where possible.\n\n"

        "GLOBAL-TO-INDIA TRANSMISSION: Assess relevant paths such as US inflation -> Fed expectations -> yields/USD -> "
        "foreign flows/USDINR -> Indian equities; oil supply shock -> crude/freight/inflation -> Indian energy and "
        "fuel-sensitive companies; China stimulus -> industrial demand/commodities -> metals; global credit stress -> "
        "risk aversion/funding -> financials. These are hypotheses, not predetermined outcomes. Consider surprise "
        "versus expectations and whether market pricing confirms the thesis.\n\n"

        "IMPORTANT SAFETY / QUALITY RULES:\n"
        "- Never fabricate current prices, price changes, volume, option-chain data, FII/DII flows, event times, "
        "article text, source URLs, corroboration, earnings impacts or stock exposure.\n"
        "- No personalized financial advice, guaranteed predictions or automatic trade signals.\n"
        "- Clearly distinguish facts supplied by headlines from analytical inference.\n"
        "- A stock affected by a story is a watchlist candidate, not a buy/sell recommendation.\n"
        "- Keep the analysis useful and concise, but include the required fields for every consolidated event.\n\n"
        f"SUPPLIED ARTICLES ({len(selected)} of {len(articles)} fetched articles; consolidate duplicates/related coverage):\n"
        f"{headlines}"
    )


def ai_analysis(articles):
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return (
            "AI analysis skipped: HF_TOKEN is not configured. Article links are included below. "
            "Set HF_TOKEN and optionally HF_MODEL to enable consolidated event analysis."
        )

    model = (os.getenv("HF_MODEL") or "openai/gpt-oss-120b:cheapest").strip()
    prompt = build_analysis_prompt(articles)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful global macro and Indian equity-market news analyst. "
                    "Consolidate duplicate coverage into event-level reports, preserve all supporting links, "
                    "map only credible stock exposures, and explicitly label uncertainty."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(os.getenv("HF_MAX_TOKENS", "10000")),
        "temperature": 0.15,
    }

    try:
        response = requests.post(
            HF_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=int(os.getenv("AI_TIMEOUT", "120")),
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        try:
            detail = (exc.response.text or "").strip()[:900] if exc.response is not None else ""
        except Exception:
            detail = ""
        return (
            f"AI analysis failed for model '{model}' (HTTP {status}).\n"
            f"Provider response: {detail or 'No response detail supplied.'}\n"
            "Headlines and links are still included. Check HF_TOKEN permissions, model/provider availability, "
            "Inference Providers credits/quota, HF_MODEL and provider limits."
        )
    except Exception as exc:
        return (
            f"AI analysis failed for model '{model}': {type(exc).__name__}: {exc}.\n"
            "Headlines and links are still included. Check network access, HF_TOKEN, model/provider availability and quota."
        )


def grouped_articles(articles):
    return {
        "Tier 1 — Official / Primary": [
            a for a in articles if a["source_class"] == "Tier 1 — Official / Primary"
        ],
        "Tier 2 — Established Reporting": [
            a for a in articles if a["source_class"] == "Tier 2 — Established Reporting"
        ],
        "Needs Validation — Other Source": [
            a for a in articles if a["source_class"] == "Needs Validation — Other Source"
        ],
    }


def article_html(a):
    title = html.escape(a["title"])
    link = html.escape(a["link"], quote=True)
    return (
        f'<li style="margin:0 0 12px 0"><a href="{link}"><b>{title}</b></a><br>'
        f'<small>Source: {html.escape(a["source"])} | Published: {html.escape(a["published"])}</small><br>'
        f'<small>Publisher class: {html.escape(a["source_class"])}</small></li>'
    )


def next_scheduled_ist(now):
    # Preserve the existing hourly schedule at minute 47 IST.
    candidate = now.replace(minute=47, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(hours=1)
    return candidate


def make_report(articles, analysis):
    now = now_ist()
    next_run = next_scheduled_ist(now)
    groups = grouped_articles(articles)

    lines = [
        f"GLOBAL + INDIA MARKET INTELLIGENCE | {now.strftime('%d %b %Y')} | "
        f"{now.strftime('%I:%M %p IST')} (UTC+05:30)",
        f"Report generated: {now.strftime('%d %b %Y, %I:%M %p IST')} | "
        f"Next scheduled run (approx.): {next_run.strftime('%d %b %Y, %I:%M %p IST')}.",
        "Workflow/scheduler delays are possible. Headlines are not a substitute for original articles or live market data.",
        f"Fetched articles: {len(articles)} | "
        f"Official-source publisher class: {len(groups['Tier 1 — Official / Primary'])} | "
        f"Established-reporting publisher class: {len(groups['Tier 2 — Established Reporting'])} | "
        f"Other publishers requiring validation: {len(groups['Needs Validation — Other Source'])}",
        "",
        "CONSOLIDATED GLOBAL + INDIAN NEWS ANALYSIS",
        "-" * 60,
        analysis,
        "",
        "SUPPORTING ARTICLE INDEX (raw fetched headlines; AI analysis may consolidate related stories)",
    ]

    for group_name in (
        "Tier 1 — Official / Primary",
        "Tier 2 — Established Reporting",
        "Needs Validation — Other Source",
    ):
        lines.extend(["", group_name, "-" * 55])
        if not groups[group_name]:
            lines.append("No headlines in this publisher category.")
        for a in groups[group_name]:
            lines.extend([
                f"• {a['title']}",
                f"  Publisher: {a['source']} | Published: {a['published']}",
                f"  Link: {a['link']}",
            ])

    lines.extend([
        "",
        "IMPORTANT INTERPRETATION NOTES",
        "Publisher source class is not the same as event-impact tier.",
        "Google News RSS publisher labels and timestamps can be incomplete or inaccurate; verify the original article.",
        "AI consolidation and stock mapping are analytical outputs, not verified live market data or trade signals.",
        "Check source articles, current price/volume, liquidity, index/sector context and risk before trading.",
    ])
    plain = "\n".join(lines)

    # HTML email keeps the analysis readable and includes clickable raw-source links.
    parts = [
        '<html><body style="font-family:Arial,sans-serif;line-height:1.45;color:#202124">',
        f'<h2>Global + India Market Intelligence | {now.strftime("%d %b %Y")} | '
        f'{now.strftime("%I:%M %p IST")} (UTC+05:30)</h2>',
        f'<p><b>Generated:</b> {now.strftime("%d %B %Y, %I:%M %p IST")}<br>'
        f'<b>Next scheduled run (approx.):</b> {next_run.strftime("%d %B %Y, %I:%M %p IST")}. '
        'Scheduled runs may be delayed.</p>',
        f'<p><b>Fetched articles:</b> {len(articles)} | '
        f'<b>Official publisher class:</b> {len(groups["Tier 1 — Official / Primary"])} | '
        f'<b>Established reporting:</b> {len(groups["Tier 2 — Established Reporting"])} | '
        f'<b>Other/validation:</b> {len(groups["Needs Validation — Other Source"])}</p>',
        '<h3>1. Consolidated Global + Indian News Analysis</h3>',
        '<p>Related coverage should be consolidated into one event report, while distinct events remain separate. '
        'Each event should include global and Indian impacts, affected sectors, stock-by-stock reasoning, source links, '
        'validation, confidence and confirmation/invalidation conditions.</p>',
        '<pre style="white-space:pre-wrap;font-family:Arial,sans-serif">'
        + html.escape(analysis) + '</pre>',
        '<h3>2. Supporting Article Index — Official / Primary Publisher Class</h3><ul>'
        + ''.join(article_html(a) for a in groups['Tier 1 — Official / Primary']) + '</ul>',
        '<h3>3. Supporting Article Index — Established Reporting</h3><ul>'
        + ''.join(article_html(a) for a in groups['Tier 2 — Established Reporting']) + '</ul>',
        '<h3>4. Supporting Article Index — Other Publishers / Validate</h3>',
        '<p>These publisher labels are not a finding that the article is false. Validate claims against primary sources '
        'or independent established reporting.</p><ul>'
        + ''.join(article_html(a) for a in groups['Needs Validation — Other Source']) + '</ul>',
        '<h3>Trading risk checklist</h3><ul>'
        '<li>Verify the original article and event/publication timestamps.</li>'
        '<li>Check whether the news is new or already priced in.</li>'
        '<li>Confirm price action, volume, liquidity and sector/index context.</li>'
        '<li>Review counterarguments and define risk before any trade.</li>'
        '<li>Do not treat a stock watchlist or directional bias as an automatic trade signal.</li>'
        '</ul>',
        '<p><small>Informational market intelligence only; automated analysis may be incomplete. '
        'No personalized investment advice or guaranteed forecasts.</small></p>',
        '</body></html>',
    ]
    return plain, ''.join(parts)


def send_email(subject, plain_body, html_body):
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

    if articles:
        analysis = ai_analysis(articles)
    else:
        analysis = (
            "No new unseen headlines found since the previous successful run. This does not mean there is no "
            "market-moving news. Check official exchange filings and current market data if making decisions."
        )

    plain, html_body = make_report(articles, analysis)
    now = now_ist()
    subject = f"Global + India Market Intelligence | {now.strftime('%d %b %Y')} | {now.strftime('%I:%M %p IST')}"
    print(plain)
    send_email(subject, plain, html_body)

    # Save only after successful email delivery, preserving retry behaviour.
    seen_ids.update(a["id"] for a in articles)
    save_seen(seen_ids)
    print("Email sent successfully.")


if __name__ == "__main__":
    main()
