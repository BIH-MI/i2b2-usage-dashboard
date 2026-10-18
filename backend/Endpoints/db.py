import os
from psycopg_pool import ConnectionPool
from contextlib import contextmanager



def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value



POSTGRES_PW = require_env("POSTGRES_PW")
PORT_DB = require_env("PORT_DB")
DB_USER = require_env("DB_USER")
DB_HOST = require_env("DB_HOST")
DB_NAME = require_env("DB_NAME")

#i2b2 schema names, the data schema can be different per dataset
DATA_SCHEMA = os.environ.get("I2B2_DATA_SCHEMA", "i2b2demodata")
PM_SCHEMA = os.environ.get("I2B2_PM_SCHEMA", "i2b2pm")



pool = ConnectionPool(
    kwargs={
    "host": DB_HOST,
    "port": PORT_DB,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": POSTGRES_PW,
    "connect_timeout": 5,
    "options": "-c statement_timeout=30000"
    },
    min_size = 3,
    max_size = 20,
    timeout = 10,
    max_waiting = 50,
    check = ConnectionPool.check_connection,
    open = True
    )

@contextmanager
def get_cursor():
    with pool.connection() as conn:
        with conn.cursor() as cur:
            yield cur


