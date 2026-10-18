from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from auth import get_current_user
from main import app

client = TestClient(app)

# skip the login check, every request counts as logged in
app.dependency_overrides[get_current_user] = lambda: "test-user"


# no monkeypatch here, the real sql runs against the demo database
def all_queries():
    response = client.get("/api/queries")
    assert response.status_code == 200
    rows = response.json()
    if not rows:
        pytest.skip("no queries in the database")
    return rows


def test_health_db():
    response = client.get("/health/db")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_date_range():
    response = client.get("/api/date-range")

    assert response.status_code == 200
    body = response.json()
    assert datetime.fromisoformat(body["min"]) <= datetime.fromisoformat(body["max"])


# same call as the query detail page
def test_queries_filter_by_id():
    query_id = all_queries()[0]["query_master_id"]

    response = client.get(f"/api/queries?query_master_id={query_id}")

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) >= 1
    assert all(row["query_master_id"] == query_id for row in rows)


def test_queries_filter_by_user():
    user_id = all_queries()[0]["user_id"]

    response = client.get("/api/queries", params={"user_id": user_id})

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) >= 1
    assert all(row["user_id"] == user_id for row in rows)


def test_overall_stats():
    date_range = client.get("/api/date-range").json()
    start = date_range["min"][:10]
    end = date_range["max"][:10]

    response = client.get(f"/api/overall-stats?start={start}&end={end}")

    assert response.status_code == 200
    stats = response.json()
    assert stats["query_count"] >= 1
    assert stats["user_count"] >= 1
    assert stats["concept_count"] >= 0


def test_usage_over_time():
    date_range = client.get("/api/date-range").json()
    start = date_range["min"][:10]
    end = date_range["max"][:10]

    response = client.get(f"/api/usage-over-time?start={start}&end={end}")

    assert response.status_code == 200
    months = response.json()
    assert months[0]["month"] == start[:7]
    assert months[-1]["month"] == end[:7]
    assert sum(m["active_users"] for m in months) >= 1
    assert all(m["new_users"] >= 0 and m["sessions"] >= 0 for m in months)
