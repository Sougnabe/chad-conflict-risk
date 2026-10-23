"""Client for the ACLED API (OAuth authentication and paginated download)."""

import pandas as pd
import requests


class AcledError(RuntimeError):
    """Raised when the ACLED API refuses a request."""


class AcledClient:
    """Downloads ACLED events for one country and period.

    Credentials are passed in (read from .env by the caller) and never stored on disk.
    """

    TOKEN_URL = "https://acleddata.com/oauth/token"
    API_URL = "https://acleddata.com/api/acled/read"

    def __init__(self, username: str, password: str, page_size: int = 5000,
                 session: requests.Session | None = None):
        if not username or not password:
            raise ValueError("ACLED username and password are required")
        self._username = username
        self._password = password
        self._page_size = page_size
        self._session = session or requests.Session()
        self._token: str | None = None

    def _authenticate(self) -> str:
        resp = self._session.post(self.TOKEN_URL, timeout=60, data={
            "username": self._username, "password": self._password,
            "grant_type": "password", "client_id": "acled"})
        if resp.status_code != 200:
            raise AcledError(f"Authentication failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json()["access_token"]

    def fetch(self, country: str, start_date: str, end_date: str) -> pd.DataFrame:
        """All events for `country` between two dates (YYYY-MM-DD), across result pages."""
        self._token = self._token or self._authenticate()
        headers = {"Authorization": f"Bearer {self._token}"}
        rows, page = [], 1
        while True:
            params = {"_format": "json", "country": country,
                      "event_date": f"{start_date}|{end_date}",
                      "event_date_where": "BETWEEN", "limit": self._page_size, "page": page}
            resp = self._session.get(self.API_URL, headers=headers, params=params, timeout=120)
            if resp.status_code != 200:
                raise AcledError(f"API error, page {page} ({resp.status_code}): "
                                 f"{resp.text[:300]}")
            data = resp.json().get("data", [])
            rows.extend(data)
            if len(data) < self._page_size:
                break
            page += 1
        return pd.DataFrame(rows)
