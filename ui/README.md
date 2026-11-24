# Ledger research workspace

Ledger is a private Streamlit workspace for researching uploaded annual reports
and SEC filings. It supports Supabase Auth through the API and bearer-token mode
for local development. Documents are uploaded to the authenticated API, and only
processed reports can be selected for a query.

## Run locally

Start the API, then launch the UI from the repository root:

```bash
uvicorn agent.server.app:app --port 8000
streamlit run ui/app.py --server.port 8501
```

The API URL defaults to `http://127.0.0.1:8000`. Set `API_URL` or change it in
the sidebar connection settings to use a hosted API. In token mode, enter the
workspace access token in the sidebar, or provide `FINAGENT_API_TOKEN` to the
Streamlit process. Supabase mode displays the email and password sign-in form;
the UI sends credentials to the API auth proxy and keeps the returned session
in Streamlit session state. A development-only API URL override is disabled by
default and can be enabled with `FINAGENT_ALLOW_API_OVERRIDE=1`. Do not enable
it in a deployed UI; server-provided bearer tokens are never forwarded to a
different URL.

## Research flow

- Upload PDF filings and track their processing status.
- Select processed reports to scope a question to those sources.
- Review the answer status, page citations, table coordinates, quoted evidence,
  and any calculation details returned by the API.
- Review paused decisions and continue the associated analysis. The resumed
  answer and citations are stored with the matching session.
- Reopen recent results during the current Streamlit session.

The UI is a thin client. Authentication, document ownership, PDF processing,
retrieval, answer generation, and review governance are enforced by the API and
worker. Visual styles live in `ui/styles.css`.
