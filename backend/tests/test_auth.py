from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import pytest
from fastapi.testclient import TestClient

import users
from auth import JWT_SECRET, COOKIE_NAME, create_token, get_current_user
from main import app

# https, the session cookie is only sent over https
client = TestClient(app, base_url="https://testserver")


# two fake dashboard users, the real users file is never read or written
@pytest.fixture
def fake_users(monkeypatch):
    saved = dict(users.USERS)
    users.USERS.clear()
    users.USERS.update({
        "admin-user": {"hash": bcrypt.hashpw(b"admin-password", bcrypt.gensalt()).decode(), "role": "admin"},
        "plain-user": {"hash": bcrypt.hashpw(b"user-password", bcrypt.gensalt()).decode(), "role": "user"},
    })
    monkeypatch.setattr(users, "save_users", lambda _users: None)
    client.cookies.clear()  # the client keeps cookies from earlier tests, e.g. after a login
    yield
    users.USERS.clear()
    users.USERS.update(saved)


# the real login check, test_api.py replaces it for its own tests
@pytest.fixture
def real_login(monkeypatch):
    monkeypatch.delitem(app.dependency_overrides, get_current_user, raising=False)


def logged_in_as(monkeypatch, username):
    monkeypatch.setitem(app.dependency_overrides, get_current_user, lambda: username)


def session_cookie(token):
    return {"Cookie": f"{COOKIE_NAME}={token}"}


# login, session and logout

def test_login_sets_session_cookie(fake_users):
    response = client.post("/api/login", json={"username": "plain-user", "password": "user-password"})

    assert response.status_code == 200
    assert response.json() == {"username": "plain-user"}
    assert COOKIE_NAME in response.cookies


def test_login_wrong_password(fake_users):
    response = client.post("/api/login", json={"username": "plain-user", "password": "wrong-password"})

    assert response.status_code == 401
    assert COOKIE_NAME not in response.cookies


def test_login_unknown_user(fake_users):
    response = client.post("/api/login", json={"username": "nobody", "password": "user-password"})

    assert response.status_code == 401


def test_me_with_valid_token(fake_users, real_login):
    response = client.get("/api/me", headers=session_cookie(create_token("plain-user")))

    assert response.status_code == 200
    assert response.json() == {"username": "plain-user", "role": "user"}


def test_me_without_cookie(fake_users, real_login):
    response = client.get("/api/me")

    assert response.status_code == 401


def test_me_with_expired_token(fake_users, real_login):
    expired = datetime.now(timezone.utc) - timedelta(minutes=1)
    token = jwt.encode({"sub": "plain-user", "exp": expired}, JWT_SECRET, algorithm="HS256")

    response = client.get("/api/me", headers=session_cookie(token))

    assert response.status_code == 401


def test_me_with_token_signed_by_another_secret(fake_users, real_login):
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    token = jwt.encode({"sub": "admin-user", "exp": expires}, "another-secret-that-is-at-least-32-bytes", algorithm="HS256")

    response = client.get("/api/me", headers=session_cookie(token))

    assert response.status_code == 401


def test_me_for_deleted_user(fake_users, real_login):
    token = create_token("plain-user")
    del users.USERS["plain-user"]

    response = client.get("/api/me", headers=session_cookie(token))

    assert response.status_code == 401


def test_logout_deletes_session_cookie(fake_users):
    response = client.post("/api/logout")

    assert response.status_code == 200
    assert "max-age=0" in response.headers["set-cookie"].lower()


# admin routes

def test_admin_routes_forbidden_for_user(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "plain-user")

    assert client.get("/api/admin/users").status_code == 403
    assert client.post("/api/admin/users", json={"username": "new-user", "password": "new-password", "role": "user"}).status_code == 403
    assert client.patch("/api/admin/users/admin-user", json={"role": "user"}).status_code == 403
    assert client.delete("/api/admin/users/admin-user").status_code == 403
    assert "new-user" not in users.USERS


def test_admin_lists_users_without_hashes(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.get("/api/admin/users")

    assert response.status_code == 200
    assert response.json() == [{"username": "admin-user", "role": "admin"}, {"username": "plain-user", "role": "user"}]
    assert "hash" not in response.text


def test_admin_adds_user(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.post("/api/admin/users", json={"username": "new-user", "password": "new-password", "role": "user"})

    assert response.status_code == 201
    assert users.USERS["new-user"]["role"] == "user"
    assert bcrypt.checkpw(b"new-password", users.USERS["new-user"]["hash"].encode())


def test_admin_adds_existing_user(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.post("/api/admin/users", json={"username": "plain-user", "password": "new-password", "role": "admin"})

    assert response.status_code == 409
    assert users.USERS["plain-user"]["role"] == "user"


def test_admin_adds_user_with_invalid_name(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.post("/api/admin/users", json={"username": "bad name!", "password": "new-password", "role": "user"})

    assert response.status_code == 422
    assert "bad name!" not in users.USERS


def test_admin_changes_role_of_another_user(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.patch("/api/admin/users/plain-user", json={"role": "admin"})

    assert response.status_code == 200
    assert users.USERS["plain-user"]["role"] == "admin"


def test_admin_cannot_change_own_role(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.patch("/api/admin/users/admin-user", json={"role": "user"})

    assert response.status_code == 400
    assert users.USERS["admin-user"]["role"] == "admin"


def test_admin_deletes_another_user(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.delete("/api/admin/users/plain-user")

    assert response.status_code == 200
    assert "plain-user" not in users.USERS


def test_admin_cannot_delete_own_account(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.delete("/api/admin/users/admin-user")

    assert response.status_code == 400
    assert "admin-user" in users.USERS


def test_admin_deletes_unknown_user(fake_users, monkeypatch):
    logged_in_as(monkeypatch, "admin-user")

    response = client.delete("/api/admin/users/nobody")

    assert response.status_code == 404
