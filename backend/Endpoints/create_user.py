#creates a dashboard user from the terminal, needed for the first admin:
#docker compose run --rm backend python create_user.py
import getpass

from users import add_user, ROLES

username = input("Username: ").strip()
role = input(f"Role {ROLES}: ").strip()
password = getpass.getpass("Password: ")

try:
    add_user(username, password, role)
    print(f"Created {username} with role {role}")
except ValueError as e:
    print(f"Not created: {e}")
