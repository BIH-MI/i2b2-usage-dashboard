from fastapi import APIRouter, Query
from datetime import date
import queries
from models import DailyCount, DateRange, OverallStats, QueryDetail, UserDailyCount


router = APIRouter()

# min/max query date
@router.get("/date-range", response_model = DateRange )
def date_range():
    return queries.get_date_range()

@router.get("/queries", response_model = list[QueryDetail])
def query_details(
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    user_id: list[str] | None = Query(default=None),
    query_master_id: int | None = Query(default=None),
):
    return queries.get_queries(start, end, user_id, query_master_id=query_master_id)


# queries per user per day
@router.get("/queries-by-user", response_model = list[UserDailyCount])
def queries_by_user(start: date, end: date, user_id: list[str] = Query(...)):
    return queries.get_queries_by_user(start, end, user_id)

# summary stats
@router.get("/overall-stats", response_model = OverallStats)
def overall_stats(start:date, end:date, user_id: list[str] | None = Query(default=None)):
    return queries.get_overall_stats(start, end, user_id)

# queries per day
@router.get("/daily-count", response_model = list[DailyCount])
def daily_count(start:date, end:date, user_id: list[str] | None = Query(default=None)):
    return queries.get_daily_queries_count(start, end, user_id)
