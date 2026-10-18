from fastapi import APIRouter, Query
from datetime import date
import queries
from models import Groups, LoginAttemptDetail, LoginStatusCount, UsageMonth, UserActivity, UsersList

router = APIRouter()

# all user ids
@router.get("/users", response_model = list[UsersList])
def users_list():
    return queries.get_all_users()

# login status totals
@router.get("/login-status-counts", response_model=list[LoginStatusCount])
def login_status_counts():
    return queries.get_login_status_counts()

# users by status
@router.get("/login-attempts-by-status", response_model=list[LoginAttemptDetail])
def login_attempts_by_status(attempt_cd: str):
    return queries.get_login_attempts_by_status(attempt_cd)

@router.get("/user-activity", response_model=list[UserActivity])
def user_activity(start: date, end: date, user_id: list[str] | None = Query(default=None)):
    return queries.get_user_activity(start, end, user_id)

@router.get("/user-groups", response_model = list[Groups])
def user_groups():
    return queries.get_user_groups()

@router.get("/usage-over-time", response_model = list[UsageMonth])
def usage_over_time(start: date, end: date, user_id: list[str] | None = Query(default=None)):
    return queries.get_usage_over_time(start, end, user_id)
