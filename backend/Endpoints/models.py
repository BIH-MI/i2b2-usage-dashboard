from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Literal


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=64)

#admin adds a dashboard user
class NewUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$") #letters, digits, dot, underscore, hyphen
    password: str = Field(min_length=8, max_length=64)
    role: Literal["admin", "user"]

#a logged in user changes their own password
class PasswordChange(BaseModel):
    current_password: str = Field(max_length=64)
    new_password: str = Field(min_length=8, max_length=64)

#admin changes the role of an existing user
class RoleChange(BaseModel):
    role: Literal["admin", "user"]

class DashboardUser(BaseModel):
    username: str
    role: str


    
class DateRange(BaseModel):
    min: datetime
    max: datetime


class OverallStats(BaseModel):
    query_count: int
    user_count: int
    concept_count: int
    temporal_count: int
    avg_complexity: float
    avg_queries_per_user: float


class DailyCount(BaseModel):
    day: date
    count: int

class UsersList(BaseModel):
    user_id: str

#queries per user per day
class UserDailyCount(BaseModel):
    user_id: str
    day: date
    count: int


#login attempts 
class LoginAttemptDetail(BaseModel):
    user_id: str
    entry_date: datetime

class LoginStatusCount(BaseModel):
    attempt_cd: str
    count: int


class UserActivity(BaseModel):
    user_id: str
    query_count: int
    valid_sessions: int
    activity_score: int

class QueryResult(BaseModel):
    result_type_name: str
    duration: float | None

class DateConstraint(BaseModel):
    date_from: str | None
    date_to: str | None

class PanelItem(BaseModel):
    key: str
    date_from: str | None
    date_to: str | None
    value_operator: str | None
    value_constraint: str | None
    value_unit: str | None

class Panel(BaseModel):
    excluded: bool
    occurrences: int
    items: list[PanelItem]

class QueryDetail(BaseModel):
    query_master_id: int
    name: str
    user_id: str
    status_name: str
    start_date: datetime
    duration: float | None
    concepts: list[str]
    included_concepts: list[str]
    excluded_concepts: list[str]
    num_constraints: int
    date_constraint_values: list[DateConstraint]
    panels: list[Panel]
    temporal_query_constraints: int
    complexity_score: int
    results: list[QueryResult]

class Groups(BaseModel):
    user_id: str
    group_name: str


class UsageMonth(BaseModel):
    month: str
    new_users: int
    active_users: int
    sessions: int
