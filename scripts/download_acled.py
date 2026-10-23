"""
Step 1 — Download all ACLED events for Chad (2015-01-01 -> 2025-10-02).

Requires ACLED_USERNAME and ACLED_PASSWORD in .env (see .env.example).

Usage: python scripts/download_acled.py
"""

import os
import sys

import _bootstrap  # noqa: F401  (makes chadrisk importable)
from dotenv import load_dotenv

from chadrisk.config import PATHS
from chadrisk.data.acled import AcledClient, AcledError

load_dotenv(PATHS.root / ".env")
try:
    client = AcledClient(os.getenv("ACLED_USERNAME"), os.getenv("ACLED_PASSWORD"))
    events = client.fetch("Chad", "2015-01-01", "2025-10-02")
except (ValueError, AcledError) as error:
    sys.exit(f"ACLED download failed: {error}")
if events.empty:
    sys.exit("No events received: check the parameters against the ACLED API guide.")

PATHS.raw.mkdir(parents=True, exist_ok=True)
events.to_csv(PATHS.acled_raw, index=False)
print(f"{len(events)} events saved to {PATHS.acled_raw}")
print(f"Period: {events['event_date'].min()} -> {events['event_date'].max()}")
