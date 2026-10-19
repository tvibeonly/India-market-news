import os
import re
import time
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

# Google News RSS -> AI analysis (Hugging Face model chain by default) -> email -> seen-state.
# Configure email credentials and HF_TOKEN through GitHub Actions Secrets.
IST = ZoneInfo("Asia/Kolkata")
RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

SEARCHES = [
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
    'Federal Reserve FOMC US inflation employment Treasury yields dollar markets when:1d',
    'site:bls.gov CPI employment site:bea.gov GDP PCE United States when:7d',
    'ECB Bank of England Bank of Japan China PBOC central bank decisions when:1d',
    'China economy stimulus manufacturing property trade tariffs markets when:1d',
    'IMF World Bank global growth financial stability sovereign debt when:7d',
    'Middle East geopolitics oil shipping sanctions crude prices when:1d',
    'Russia Ukraine sanctions war trade global markets when:1d',
    'global tariffs trade restrictions export controls supply chain markets when:1d',
    'OPEC oil supply EIA inventories IEA energy markets when:1d',
    'global banking stress credit markets volatility currencies when:1d',
    'Reuters global markets India stocks economy when:1d',
    'Bloomberg global markets central banks commodities when:1d',
    'Financial Times geopolitics trade economy markets when:1d',
    'CNBC world markets global economy when:1d',
    'CNBC-TV18 Indian markets stocks when:1d',
    'Business Standard India economy stocks markets when:1d',
    'Economic Times Markets India stocks when:1d',
]

MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "60"))
# Kept modest so one request fits Groq's free-tier tokens-per-minute limit.
MAX_AI_ARTICLES = int(os.getenv("MAX_AI_ARTICLES", "15"))
STATE_FILE = os.getenv("STATE_FILE", "seen_news.json")
HF_URL = "https://router.huggingface.co/v1/chat/completions"
# The script needs the complete OpenAI-compatible endpoint, not just the base URL.
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/v1/chat/completions"
).rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "25"))

# Hugging Face model chain, best first. Each is tried in order until one answers.
# Override with the HF_MODELS repo variable (comma-separated) or a single HF_MODEL.
DEFAULT_HF_MODELS = [
    "deepseek-ai/DeepSeek-V4-Pro-0813",   # flagship reasoning model
    "moonshotai/Kimi-K3",                  # strong long-context reasoning
    "zai-org/GLM-5.3",                     # strong general/analytical model
    "Qwen/Qwen3.8-27B",                    # smaller, fast, dependable
    "deepseek-ai/DeepSeek-V4.1-Flash",     # cheap fast fallback
]

# Groq (free tier, OpenAI-compatible). Models tried best first; unavailable ones are skipped.
# Override with the GROQ_MODELS repo variable (comma-separated).
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

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
    for query_index, query in enumerate(SEARCHES, start=1):
        print(f"Fetching RSS query {query_index}/{len(SEARCHES)}: {query}", flush=True)
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
    print(f"Preparing AI analysis for {len(selected)} articles (from {len(articles)} fetched).", flush=True)
    headlines = "\n".join(
        f"{i}. Title: {a['title']} | Publisher: {a['source']} | "
        f"Publisher class: {a['source_class']} | Published (RSS metadata, IST): {a['published']} | "
        f"URL: {a['link']} | Search context: {a['query_context']}"
        for i, a in enumerate(selected, 1)
    )

    return (
        "Prepare a concise, event-based global-and-Indian market-intelligence report for an intraday equity trader. "
        "DO NOT write a general description of RSS feeds, publishers, or the types of companies appearing in the feed. "
        "Analyse the underlying news events instead. If the supplied headlines do not contain enough event facts, "
        "say so and put the item in the validation queue rather than producing a generic feed summary. "
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
        "OUTPUT LENGTH: Prefer the 5–10 most market-relevant distinct events. Do not give every RSS headline its own report. "
        "If fewer than five events have adequate evidence, report fewer and explain the data gap. Keep each event focused.\n\n"
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


def hf_model_chain():
    """Ordered list of Hugging Face models to try (best first)."""
    raw = (os.getenv("HF_MODELS") or os.getenv("HF_MODEL") or "").strip()
    models = [m.strip() for m in raw.split(",") if m.strip()] or list(DEFAULT_HF_MODELS)
    # ':fastest' lets the router pick a live provider; keep user-supplied suffixes untouched.
    return [m if ":" in m else f"{m}:fastest" for m in models]


def clean_model_output(text):
    """Strip reasoning-model <think> blocks so only the report is emailed."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def call_chat(url, headers, payload):
    # Retry on rate limits (HTTP 429), waiting as long as the provider asks.
    for attempt in range(3):
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=int(os.getenv("AI_TIMEOUT", "300")),
        )
        if response.status_code == 429 and attempt < 2:
            try:
                wait = float(response.headers.get("retry-after", "30"))
            except ValueError:
                wait = 30.0
            wait = min(max(wait, 5.0), 90.0)
            print(f"Rate limited (429); waiting {wait:.0f}s then retrying.", flush=True)
            time.sleep(wait)
            continue
        break
    response.raise_for_status()
    content = clean_model_output(response.json()["choices"][0]["message"]["content"] or "")
    if not content:
        raise ValueError("The AI provider returned an empty response")
    return content


def ai_analysis(articles):
    """Hugging Face model chain by default; local Ollama if AI_PROVIDER=ollama."""
    if os.getenv("GROQ_API_KEY", "").strip():
        default_provider = "groq"
    elif os.getenv("HF_TOKEN", "").strip():
        default_provider = "huggingface"
    else:
        default_provider = "ollama"
    provider = (os.getenv("AI_PROVIDER") or default_provider).strip().lower()
    prompt = build_analysis_prompt(articles)
    base_payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful global macro and Indian equity-market news analyst. "
                    "Consolidate duplicate coverage into event-level reports, preserve all supporting links, "
                    "map only credible stock exposures, and explicitly label uncertainty. "
                    "Output only the final report, with no preamble or reasoning transcript."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(os.getenv("AI_MAX_TOKENS", os.getenv("HF_MAX_TOKENS", "6000"))),
        "temperature": 0.15,
    }

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            return (
                "AI analysis skipped: GROQ_API_KEY is not configured. "
                "Article links are included below."
            )
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        raw = (os.getenv("GROQ_MODELS") or os.getenv("GROQ_MODEL") or "").strip()
        groq_models = [m.strip() for m in raw.split(",") if m.strip()] or list(DEFAULT_GROQ_MODELS)
        attempts = [(GROQ_URL, headers, m) for m in groq_models]
    elif provider == "ollama":
        model = (os.getenv("OLLAMA_MODEL") or "qwen2.5:3b").strip()
        attempts = [(OLLAMA_URL, {"Content-Type": "application/json"}, model)]
    elif provider in ("huggingface", "hf"):
        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            return (
                "AI analysis skipped: HF_TOKEN is not configured for the Hugging Face provider. "
                "Article links are included below."
            )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        attempts = [(HF_URL, headers, m) for m in hf_model_chain()]
    else:
        return (
            f"AI analysis skipped: unsupported AI_PROVIDER '{provider}'. "
            "Use 'huggingface' or 'ollama'. Article links are included below."
        )

    errors = []
    for url, headers, model in attempts:
        payload = dict(base_payload, model=model)
        if provider == "groq":
            # Keep reasoning short/hidden so the token budget goes to the report.
            if "gpt-oss" in model:
                payload["reasoning_effort"] = "low"
            elif "qwen3" in model:
                payload["reasoning_format"] = "hidden"
        try:
            print(f"AI provider={provider}; model={model}; endpoint={url}", flush=True)
            content = call_chat(url, headers, payload)
            print(f"AI analysis succeeded with model {model}.", flush=True)
            return f"[Model: {model}]\n\n{content}"
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            try:
                detail = (exc.response.text or "").strip()[:300] if exc.response is not None else ""
            except Exception:
                detail = ""
            errors.append(f"- {model}: HTTP {status} {detail}")
            print(f"Model {model} failed (HTTP {status}); trying next.", flush=True)
        except Exception as exc:
            errors.append(f"- {model}: {type(exc).__name__}: {exc}")
            print(f"Model {model} failed ({type(exc).__name__}); trying next.", flush=True)

    if provider == "groq":
        help_text = (
            "Check GROQ_API_KEY, daily/per-minute limits on console.groq.com (Limits page), "
            "and the GROQ_MODELS list; model names change over time."
        )
    elif provider == "ollama":
        help_text = "Check that Ollama is running, the model was downloaded, and OLLAMA_URL/OLLAMA_MODEL are correct."
    else:
        help_text = (
            "Check HF_TOKEN permissions (Inference Providers), credits/quota, and the HF_MODELS list. "
            "Model availability changes; see https://router.huggingface.co/v1/models"
        )
    return (
        f"AI analysis failed for provider '{provider}'. Models tried:\n"
        + "\n".join(errors)
        + f"\n{help_text}\nHeadlines and links are still included below."
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
    # Workflow runs once a day at 02:30 UTC = 08:00 IST (before the 09:15 market open).
    candidate = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
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

    print(f"RSS collection complete: {len(articles)} unseen articles selected.", flush=True)
    if articles:
        print("Starting AI event consolidation and impact analysis; this step may take several minutes.", flush=True)
        analysis = ai_analysis(articles)
        print("AI analysis complete.", flush=True)
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
