"""
Download all ACLED events for Chad (2015-01-01 -> 2025-10-02) through the ACLED API.

Usage:
    1. Copy .env.example to .env and fill in your myACLED credentials.
    2. pip install requests python-dotenv pandas
    3. python scripts/download_acled.py

If authentication fails, check the token URL and parameter names below
against the myACLED "API guide".
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "data" / "raw" / "acled_chad_2015_2025.csv"

TOKEN_URL = "https://acleddata.com/oauth/token"
API_URL = "https://acleddata.com/api/acled/read"

COUNTRY = "Chad"
START_DATE = "2015-01-01"
END_DATE = "2025-10-02"
PAGE_SIZE = 5000


def get_token(username: str, password: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        data={
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
        },
        timeout=60,
    )
    if resp.status_code != 200:
        sys.exit(f"Authentication failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["access_token"]


def fetch_all(token: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"}
    rows, page = [], 1
    while True:
        params = {
            "_format": "json",
            "country": COUNTRY,
            "event_date": f"{START_DATE}|{END_DATE}",
            "event_date_where": "BETWEEN",
            "limit": PAGE_SIZE,
            "page": page,
        }
        resp = requests.get(API_URL, headers=headers, params=params, timeout=120)
        if resp.status_code != 200:
            sys.exit(f"API error, page {page} ({resp.status_code}): {resp.text[:300]}")
        data = resp.json().get("data", [])
        print(f"Page {page}: {len(data)} events")
        if not data:
            break
        rows.extend(data)
        if len(data) < PAGE_SIZE:
            break
        page += 1
    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv(ROOT / ".env")
    username = os.getenv("ACLED_USERNAME")
    password = os.getenv("ACLED_PASSWORD")
    if not username or not password:
        sys.exit("ACLED_USERNAME and ACLED_PASSWORD must be set in the .env file")

    token = get_token(username, password)
    df = fetch_all(token)
    if df.empty:
        sys.exit("No events received: check the parameters against the API guide.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n{len(df)} events saved to {OUT_FILE}")
    print(f"Period: {df['event_date'].min()} -> {df['event_date'].max()}")
    print(f"Countries: {df['country'].unique().tolist()}")


if __name__ == "__main__":
    main()
