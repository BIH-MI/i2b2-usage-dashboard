from contextlib import contextmanager
from datetime import date
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import bcrypt
import queries
import users
from auth import get_current_user
from main import app

client = TestClient(app)

# skip the login check, every request counts as logged in
app.dependency_overrides[get_current_user] = lambda: "test-user"


def test_api_needs_login(monkeypatch):
    monkeypatch.delitem(app.dependency_overrides, get_current_user) # real login check for this test only

    response = client.get("/api/date-range")

    assert response.status_code == 401


def test_usage_over_time_fills_empty_months(monkeypatch):
    results = iter([
        [(date(2025, 1, 1), 2)],            # new users
        [(date(2025, 1, 1), 5), (date(2025, 3, 1), 1)],  # active users
        [],                                  # sessions
    ])
    cursor = MagicMock()
    cursor.fetchall.side_effect = lambda: next(results)

    @contextmanager
    def fake_cursor():
        yield cursor

    monkeypatch.setattr(queries, "get_cursor", fake_cursor)

    response = client.get("/api/usage-over-time?start=2025-01-15&end=2025-03-10")

    assert response.status_code == 200
    assert response.json() == [
        {"month": "2025-01", "new_users": 2, "active_users": 5, "sessions": 0},
        {"month": "2025-02", "new_users": 0, "active_users": 0, "sessions": 0},
        {"month": "2025-03", "new_users": 0, "active_users": 1, "sessions": 0},
    ]


# a fake dashboard user with the password "old-password", nothing is written to /data
def fake_user(monkeypatch):
    hashed = bcrypt.hashpw(b"old-password", bcrypt.gensalt()).decode()
    monkeypatch.setitem(users.USERS, "test-user", {"hash": hashed, "role": "user"})
    monkeypatch.setattr(users, "save_users", lambda _users: None)


def test_change_password(monkeypatch):
    fake_user(monkeypatch)

    response = client.post("/api/change-password", json={"current_password": "old-password", "new_password": "new-password"})

    assert response.status_code == 200
    assert bcrypt.checkpw(b"new-password", users.USERS["test-user"]["hash"].encode())
    assert users.USERS["test-user"]["role"] == "user"


def test_change_password_wrong_current(monkeypatch):
    fake_user(monkeypatch)

    response = client.post("/api/change-password", json={"current_password": "wrong", "new_password": "new-password"})

    assert response.status_code == 400
    assert bcrypt.checkpw(b"old-password", users.USERS["test-user"]["hash"].encode())


def test_change_password_too_short(monkeypatch):
    fake_user(monkeypatch)

    response = client.post("/api/change-password", json={"current_password": "old-password", "new_password": "short"})

    assert response.status_code == 422


def test_change_password_same_as_current(monkeypatch):
    fake_user(monkeypatch)

    response = client.post("/api/change-password", json={"current_password": "old-password", "new_password": "old-password"})

    assert response.status_code == 400
