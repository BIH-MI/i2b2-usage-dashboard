import os
from pathlib import Path

import pytest

# no defaults, the connection comes from the environment or from the project .env
SETTINGS = {
    "INTEGRATION_DB_HOST": "DB_HOST",
    "INTEGRATION_DB_PORT": "PORT_DB",
    "INTEGRATION_DB_USER": "DB_USER",
    "INTEGRATION_DB_NAME": "DB_NAME",
    "I2B2_PG_PASS": "POSTGRES_PW",
}


def read_env_file():
    env_file = Path(__file__).resolve().parents[3] / ".env"
    values = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip('"')
    return values


env_file_values = read_env_file()
missing = []
for name, backend_name in SETTINGS.items():
    value = os.environ.get(name) or env_file_values.get(name)
    if value:
        # os.environ[...] and not setdefault, tests/conftest.py already set fake values
        os.environ[backend_name] = value
    else:
        missing.append(name)


@pytest.fixture(scope="session", autouse=True)
def need_db():
    if missing:
        pytest.skip("not set: " + ", ".join(missing))
    import db
    try:
        with db.get_cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        pytest.skip("postgres not running")
