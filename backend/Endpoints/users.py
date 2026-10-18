import json
import os
import re
import threading

import bcrypt

USERS_FILE = "/data/users.json"
ROLES = ("admin", "user")
#same limits as NewUserRequest in models.py
MIN_PASSWORD_LENGTH = 8
MAX_USERNAME_LENGTH = 64
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$") #letters, digits, dot, underscore, hyphen

_lock = threading.Lock() #only one request may change the file at a time


def load_users():
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {} #no users yet


#writes a temporary file first and then swaps it in, a crash cannot leave half a file
def save_users(users):
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)

USERS = load_users() #dict {"user": {"hash": "...", "role": "admin"}}


def check_password(password):
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password needs at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode()) > 72: #bcrypt cannot take more than 72 bytes
        raise ValueError("Password too long")


def add_user(username, password, role):
    if not username:
        raise ValueError("Username is empty")
    if len(username) > MAX_USERNAME_LENGTH:
        raise ValueError(f"Username can have at most {MAX_USERNAME_LENGTH} characters")
    if not USERNAME_PATTERN.match(username):
        raise ValueError("Username may only contain letters, digits, dot, underscore and hyphen")
    if role not in ROLES:
        raise ValueError("Unknown role")
    check_password(password)
    with _lock:
        if username in USERS:
            raise ValueError("User already exists")
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        USERS[username] = {"hash": hashed, "role": role}
        save_users(USERS)


def set_role(username, role):
    if role not in ROLES:
        raise ValueError("Unknown role")
    with _lock:
        if username not in USERS:
            raise ValueError("Unknown user")
        USERS[username]["role"] = role
        save_users(USERS)


#new hash with a new salt, the role stays
def set_password(username, password):
    check_password(password)
    with _lock:
        if username not in USERS:
            raise ValueError("Unknown user")
        USERS[username]["hash"] = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        save_users(USERS)


def delete_user(username):
    with _lock:
        if username not in USERS:
            raise ValueError("Unknown user")
        del USERS[username]
        save_users(USERS)
