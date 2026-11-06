"""Newly constructed dataset, step 1: mine recent Moodle issues with story points.

Source: Moodle's public Jira (tracker.moodle.org, now hosted at moodle.atlassian.net), anonymous
REST API v2 enhanced search (/rest/api/2/search/jql, token pagination). No login is needed.

Query: every MDL issue that has a story point (custom field 10043, "Story Points") and was created
after 2016-08-08, i.e. after the GPT2SP/Deep-SE benchmark snapshot, oldest first.

Raw API pages are stored unchanged in raw/page_XXX.json, together with raw/MANIFEST.json (query,
fields, page count, issue count, UTC collection time). preprocess.py applies the filtering.

Usage: python new_data/moodle_recent/collect.py
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
URL = "https://moodle.atlassian.net/rest/api/2/search/jql"
JQL = 'project = MDL AND cf[10043] is not EMPTY AND created > "2016-08-08" ORDER BY created ASC'
FIELDS = ("summary,description,customfield_10043,created,resolutiondate,updated,status,resolution,"
          "issuetype,priority,components")
PAGE_SIZE = 100


def main():
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    for old in raw.glob("page_*.json"):
        old.unlink()
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "comp8240-gpt2sp-replication"})
    token, page, total = None, 0, 0
    while True:
        params = {"jql": JQL, "fields": FIELDS, "maxResults": PAGE_SIZE}
        if token:
            params["nextPageToken"] = token
        for attempt in range(5):
            r = session.get(URL, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(int(r.headers.get("Retry-After", 2 ** attempt * 5)))
                continue
            r.raise_for_status()
            break
        data = r.json()
        (raw / f"page_{page:03d}.json").write_text(json.dumps(data))
        total += len(data.get("issues", []))
        page += 1
        print(f"page {page:3d}  issues so far {total}", flush=True)
        token = data.get("nextPageToken")
        if data.get("isLast", True) or not token:
            break
        time.sleep(0.5)  # be polite to the public server
    manifest = {"source": URL, "jql": JQL, "fields": FIELDS, "pages": page, "issues": total,
                "collected_at": datetime.now(timezone.utc).isoformat()}
    (raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
