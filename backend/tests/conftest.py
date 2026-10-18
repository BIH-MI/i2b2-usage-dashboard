import os
import tempfile

os.environ.setdefault("POSTGRES_PW", "test")
os.environ.setdefault("PORT_DB", "5432")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_NAME", "test")

# fake jwt secret, the real one only exists inside docker (/run/secrets)
secrets_dir = tempfile.mkdtemp()
with open(os.path.join(secrets_dir, "jwt_secret"), "w") as f:
    f.write("test-secret-that-is-at-least-32-bytes")
os.environ.setdefault("SECRETS_DIR", secrets_dir)
