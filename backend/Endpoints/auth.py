from users import USERS
from datetime import datetime, timedelta, timezone
import os

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request

SECRETS_DIR = os.environ.get("SECRETS_DIR", "/run/secrets") #tests point this to a temp folder
COOKIE_NAME = "session"
TOKEN_HOURS = 8

def read_secret(name):
    try:
        with open(f"{SECRETS_DIR}/{name}") as f:
            return f.read().strip()

    except FileNotFoundError:
        raise RuntimeError(f"Missing secret file: {name}")


JWT_SECRET = read_secret("jwt_secret")


def authenticate(username, password):
    if not password or username not in USERS:
        return False
    if len(password.encode()) > 72: #bcrypt cannot take more than 72 bytes, such a password is just wrong
        return False
    return bcrypt.checkpw(password.encode(), USERS[username]["hash"].encode())



#token holds the username (sub) and expiry (exp), signed with JWT_SECRET - readable but not changeable
def create_token(username):
    expires = datetime.now(timezone.utc) + timedelta(hours = TOKEN_HOURS)
    return jwt.encode({"sub": username, "exp": expires}, JWT_SECRET, algorithm="HS256")


def get_current_user(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Session expired")
    username = payload["sub"]
    if username not in USERS: #user was deleted after the token was made
        raise HTTPException(status_code=401, detail="Not logged in")
    return username


#only admins pass, everyone else gets 403. 
def require_admin(username: str = Depends(get_current_user)):
    if USERS[username]["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return username
