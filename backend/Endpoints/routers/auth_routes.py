from fastapi import APIRouter, Depends, HTTPException, Response
from auth import authenticate, create_token, get_current_user, COOKIE_NAME, TOKEN_HOURS
from models import LoginRequest, PasswordChange
from users import USERS, set_password

router = APIRouter()


@router.post("/login")
def login(body: LoginRequest, response: Response ):
    username = body.username
    password = body.password
    if not authenticate(username, password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(username)
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        secure=True, #only sent over https
        samesite="strict",
        max_age=TOKEN_HOURS * 3600,
        path="/",
    )

    return {"username": username}

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(
        COOKIE_NAME,
        path="/",
        )

    return {"status": "ok"}

#who is logged in, the frontend uses the role to show or hide the settings page
@router.get("/me")
def me(username: str = Depends(get_current_user)):
    return {"username": username, "role": USERS[username]["role"]}


#every logged in user can change their own password, the current one is asked again
@router.post("/change-password")
def change_password(body: PasswordChange, username: str = Depends(get_current_user)):
    if not authenticate(username, body.current_password):
        raise HTTPException(status_code=400, detail="Current password is wrong")
    if body.new_password == body.current_password:
        raise HTTPException(status_code=400, detail="New password must be different from the current one")
    try:
        set_password(username, body.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}
