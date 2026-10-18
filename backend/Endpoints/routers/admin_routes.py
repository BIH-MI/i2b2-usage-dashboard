from fastapi import APIRouter, Depends, HTTPException
from auth import require_admin
from models import NewUserRequest, RoleChange, DashboardUser
from users import USERS, add_user, set_role, delete_user

#every endpoint here is admin only, the check is set in main.py
router = APIRouter(prefix="/admin")


#usernames and roles only, the hashes never leave the backend
@router.get("/users", response_model=list[DashboardUser])
def list_users():
    return [{"username": name, "role": data["role"]} for name, data in sorted(USERS.items())]

@router.post("/users", response_model=DashboardUser, status_code=201)
def create_user(body: NewUserRequest):
    if body.username in USERS:
        raise HTTPException(status_code=409, detail="This username already exists")
    try:
        add_user(body.username, body.password, body.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"username": body.username, "role": body.role}


#an admin cannot change the own role, so at least one admin always stays
@router.patch("/users/{username}", response_model=DashboardUser)
def change_role(username: str, body: RoleChange, admin: str = Depends(require_admin)):
    if username not in USERS:
        raise HTTPException(status_code=404, detail="Unknown user")
    if username == admin:
        raise HTTPException(status_code=400, detail="You cannot change your own role")
    set_role(username, body.role)
    return {"username": username, "role": body.role}


#same for deleting: not the own account
@router.delete("/users/{username}")
def remove_user(username: str, admin: str = Depends(require_admin)):
    if username not in USERS:
        raise HTTPException(status_code=404, detail="Unknown user")
    if username == admin:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    delete_user(username)
    return {"status": "ok"}
