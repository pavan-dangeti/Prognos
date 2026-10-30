"""Tests for the fleet dashboard page.

The page itself carries no data, so it is served without the bearer check; every number
it shows comes through a JSON route that keeps the check. Both halves are pinned here,
along with the Content-Security-Policy that limits the page to its own script.
"""

from __future__ import annotations

import base64
import hashlib
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from prognos.stream.api import build_dashboard_state
from prognos.stream.auth import Authorizer
from prognos.stream.service import build_server
from prognos.stream.ui import CONTENT_SECURITY_POLICY, PAGE


@pytest.fixture
def dashboard() -> Iterator[str]:
    """A dashboard with authentication switched on."""
    state = build_dashboard_state()
    state.authorizer = Authorizer(token="s3cret-token-for-tests")
    httpd = build_server(state, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def status_of(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def test_the_page_is_served_with_its_policy(dashboard: str) -> None:
    with urllib.request.urlopen(f"{dashboard}/ui", timeout=5) as response:
        body = response.read().decode()
        assert response.headers["Content-Type"].startswith("text/html")
        assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert body == PAGE


def test_the_data_behind_the_page_still_needs_the_token(dashboard: str) -> None:
    assert status_of(f"{dashboard}/fleet") == 401


def test_the_policy_admits_exactly_the_inline_script() -> None:
    """If the script changes without its hash, the browser refuses to run the page."""
    (script,) = re.findall(r"<script>(.*?)</script>", PAGE, flags=re.DOTALL)
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"script-src 'sha256-{digest}'" in CONTENT_SECURITY_POLICY
    assert "default-src 'none'" in CONTENT_SECURITY_POLICY


def test_the_page_loads_nothing_from_elsewhere() -> None:
    assert "http://" not in PAGE.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in PAGE


def test_the_page_lists_every_instrument() -> None:
    from prognos.stream.window import N_CHANNELS

    state = build_dashboard_state()
    assert len(state.list_route["degraded"]()["instruments"]) == N_CHANNELS
