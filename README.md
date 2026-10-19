# Hourly Indian Stock-Market News Email

Runs in GitHub Actions, so your own computer can be off. It fetches Google News RSS headlines, optionally asks a Hugging Face Inference Provider model for a cautious impact summary, and emails the report via Gmail SMTP.

## Files
- `market_news.py`: news fetch, optional AI summary, email.
- `.github/workflows/hourly-news.yml`: hourly cloud schedule.
- `seen_news.json`: automatically created and committed to avoid repeating old articles.

## Setup
1. Create a **private** GitHub repository (recommended) and upload these files, keeping the workflow path exactly `.github/workflows/hourly-news.yml`.
2. In repository **Settings → Secrets and variables → Actions → Secrets**, add:
   - `SMTP_USER`: Gmail address that sends the report.
   - `SMTP_APP_PASSWORD`: Gmail App Password (not your normal Gmail password).
   - `EMAIL_TO`: destination email address.
   - `HF_TOKEN`: Hugging Face token with inference permissions (optional).
3. Under **Settings → Secrets and variables → Actions → Variables**, optionally add:
   - `HF_MODEL`: a model that currently supports chat completion through Hugging Face Inference Providers. Default: `Qwen/Qwen2.5-7B-Instruct:cheapest`.
   - `SMTP_HOST`: default `smtp.gmail.com`.
   - `SMTP_PORT`: default `465`.
4. Under **Settings → Actions → General → Workflow permissions**, enable **Read and write permissions** so the workflow can update `seen_news.json`.
5. Open **Actions → Hourly Indian Market News → Run workflow** to test it once.
6. Check your inbox and the workflow log. Then it will run on schedule.

## Gmail app password
Google requires 2-Step Verification to create an App Password for this approach. Visit Google Account → Security → App passwords. Use the generated app password as `SMTP_APP_PASSWORD`, not your usual password. Some work/school accounts or Advanced Protection configurations may not offer App Passwords.

## Important notes
- GitHub scheduled workflows use UTC here and may be delayed; the cron is at minute 17 to reduce top-of-hour congestion. It is hourly, not guaranteed to arrive at an exact minute.
- Public repositories may have scheduled workflows disabled after 60 days without repository activity. A private repo avoids that specific public-repo policy but free usage quotas still apply.
- Hugging Face free access is not guaranteed for every model or at every time. If the model is unavailable or quota is exhausted, the email still contains headlines and an AI error note.
- Google News RSS is an aggregator; this script analyzes headlines and metadata, not full article text.
- It sends an email each successful run, including a “no new articles” report if no new items were found.
- Do not commit secrets or tokens into code.
