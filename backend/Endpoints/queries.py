from psycopg import sql
from db import get_cursor, DATA_SCHEMA, PM_SCHEMA
from parsing import parse_query_xml

CONCEPT_TOUCH_WEIGHT = 1
CONSTRAINTS_WEIGHT = 3
TEMPORAL_WEIGHT = 5
QUERY_WEIGHT = 3
LOGIN_WEIGHT = 1


#fills {data} and {pm} in the sql with the schema names
def q(text):
    return sql.SQL(text).format(data=sql.Identifier(DATA_SCHEMA), pm=sql.Identifier(PM_SCHEMA))

#default date range when app mount
def get_date_range():
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT MIN(create_date), MAX(create_date)
        FROM {data}.qt_query_master
        WHERE request_xml IS NOT NULL"""))
        date_range = cur.fetchone()
    
    return {
        "min": date_range[0],
        "max": date_range[1]
        }

#concepts, constraints and temporal ordering, weighted
def complexity_score(parsed):
    return (
        len(parsed["item_keys"]) * CONCEPT_TOUCH_WEIGHT
        + parsed["num_constraints"] * CONSTRAINTS_WEIGHT
        + parsed["temporal_query_constraints"] * TEMPORAL_WEIGHT
    )


#overall stats shown
def get_overall_stats(start, end, user_ids=None):
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT user_id, create_date, request_xml
        FROM {data}.qt_query_master
        WHERE create_date >= %s AND create_date < %s + INTERVAL '1 day'
        AND request_xml IS NOT NULL
        AND (%s::text[] IS NULL OR user_id = ANY(%s))
        """), (start, end, user_ids, user_ids))
    #end + 1 day: create_date is a timestamp but end is just a date (midnight),
    #plain BETWEEN would read end as midnight and skip queries created later that same day
   
        rows = cur.fetchall()  

        #all i2b2 accounts, not only the ones that ran a query in this range (same list as the user filter)
        cur.execute(q("""
        SELECT COUNT(*)
        FROM {pm}.pm_user_data
        WHERE (%s::text[] IS NULL OR user_id = ANY(%s))
        """), (user_ids, user_ids))
        user_count = cur.fetchone()[0]

    query_count = len(rows)

    #users that ran at least one query in this range, for the average below
    unique_users = {user_id for user_id, create_date, request_xml in rows}

    #complexity and temporal count
    unique_concepts = set()
    temporal_count = 0
    total_complexity = 0

  
    for user_id, _, request_xml in rows:
        parsed = parse_query_xml(request_xml)

        unique_concepts.update(parsed["item_keys"])  

        if parsed["temporal_query_constraints"] > 0:
            temporal_count += 1


        total_complexity += complexity_score(parsed)

    concept_count = len(unique_concepts)
    avg_complexity = round(total_complexity/ query_count, 1 ) if query_count else 0
    avg_queries_per_user = round(query_count/ len(unique_users), 1) if unique_users else 0

    return {
        "query_count": query_count,
        "user_count": user_count,
        "concept_count" : concept_count,
        "temporal_count": temporal_count,
        "avg_complexity": avg_complexity,
        "avg_queries_per_user": avg_queries_per_user,
        }


#queries daily, for calendar heatmap, no user id needs to be returned

def get_daily_queries_count(start, end, user_ids=None):
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT create_date::date AS day,  COUNT(*)
        FROM {data}.qt_query_master
        WHERE create_date >= %s AND create_date < %s + INTERVAL '1 day'
        AND request_xml IS NOT NULL
        AND (%s::text[] IS NULL OR user_id = ANY(%s))
        GROUP BY day
        ORDER BY day """), (start, end, user_ids, user_ids))
        rows = cur.fetchall()


    result = []

    for day, count in rows:
        result.append({"day": day, "count": count})

    return  result



#queries per user per day , needs at least one user_id given
def get_queries_by_user(start, end, user_ids):
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT user_id, create_date::date AS day, COUNT(*) AS count
        FROM {data}.qt_query_master
        WHERE create_date >= %s AND create_date < %s + INTERVAL '1 day'
        AND request_xml IS NOT NULL
        AND user_id = ANY(%s)
        GROUP BY user_id, day
        ORDER BY user_id, day
        """), (start, end, user_ids))
        rows = cur.fetchall()


    result = [] 
    for user_id, day, count in rows:
        result.append({"user_id": user_id, "day": day, "count": count})
    return result


def get_all_users():
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT user_id
        FROM {pm}.pm_user_data
        ORDER BY user_id"""))
        rows = cur.fetchall()

    result = []
    for user_id, in rows:
        result.append({"user_id": user_id})
    return result



def get_login_status_counts():
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT attempt_cd, COUNT(*) AS count
        FROM {pm}.pm_user_login
        WHERE attempt_cd = UPPER(attempt_cd)
        GROUP BY attempt_cd

        UNION ALL

        SELECT 'LOCKEDOUT' AS attempt_cd, COUNT(*) AS count
        FROM {pm}.pm_project_user_params
        WHERE param_name_cd = 'LOCKEDOUT'
    """))

        rows = cur.fetchall()
    
    result = []
    for attempt_cd, count in rows:
        result.append({"attempt_cd": attempt_cd , "count":count })
    return result
    


def get_login_attempts_by_status(attempt_cd):
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT user_id, entry_date
        FROM {pm}.pm_user_login
        WHERE attempt_cd = %s

        UNION ALL
        SELECT user_id, entry_date
        FROM {pm}.pm_project_user_params
        WHERE param_name_cd = %s

        ORDER BY entry_date DESC

        
    """), (attempt_cd,attempt_cd)) 

        rows = cur.fetchall()

    result = []
    for user_id, date in rows:
        result.append({"user_id": user_id, "entry_date": date })
    return result


#weighted activity score per user, optionally restricted to a set of user_ids
def get_user_activity(start, end, user_ids=None):
    with get_cursor() as cur:

        #total queries per user in range
        cur.execute(q("""
        SELECT user_id, COUNT(*) AS query_count
        FROM {data}.qt_query_master
        WHERE create_date >= %s AND create_date < %s + INTERVAL '1 day'
        AND request_xml IS NOT NULL
        AND (%s::text[] IS NULL OR user_id = ANY(%s))
        GROUP BY user_id
        """), (start, end, user_ids, user_ids))
        query_counts = {}
        for user_id, count in cur.fetchall(): 
            query_counts[user_id] = count

        #sessions that had at least 1 query in them ( not just a login with no activity)
        cur.execute(q("""
        SELECT s.user_id, COUNT(DISTINCT s.session_id) AS valid_sessions
        FROM {pm}.pm_user_session s
        JOIN {data}.qt_query_master qm
            ON qm.user_id = s.user_id
            AND qm.create_date BETWEEN s.entry_date AND s.expired_date
            AND qm.request_xml IS NOT NULL
        WHERE s.entry_date >= %s AND s.entry_date < %s + INTERVAL '1 day'
        AND (%s::text[] IS NULL OR s.user_id = ANY(%s))
        GROUP BY s.user_id
        """), (start, end, user_ids, user_ids))
        session_counts = {}
        for user_id, count in cur.fetchall():
            session_counts[user_id] = count


    #union, a user might have queries but no matching session
    all_users = set(query_counts) | set(session_counts)
    result = []
    for user_id in all_users:
        query_count = query_counts.get(user_id, 0)
        valid_sessions = session_counts.get(user_id, 0)
        activity_score = query_count * QUERY_WEIGHT + valid_sessions * LOGIN_WEIGHT
        result.append({
            "user_id": user_id,
            "query_count": query_count,
            "valid_sessions": valid_sessions,
            "activity_score": activity_score
        })
    return result

#every query parsed
def get_queries(start=None, end=None, user_ids=None, query_master_id=None):
    with get_cursor() as cur:

        #one row per query run (query_instance), the overall/total duration
        cur.execute(q("""
        SELECT qi.query_instance_id, qm.query_master_id, qm.name, qm.user_id, st.name AS status_name,
               qi.start_date, qi.end_date, qm.request_xml
        FROM {data}.qt_query_instance qi
        JOIN {data}.qt_query_master qm ON qi.query_master_id = qm.query_master_id
        JOIN {data}.qt_query_status_type st ON qi.status_type_id = st.status_type_id
        WHERE qm.request_xml IS NOT NULL
        AND (%s::timestamp IS NULL OR qm.create_date >= %s)
        AND (%s::timestamp IS NULL OR qm.create_date < %s::timestamp + INTERVAL '1 day')
        AND (%s::int IS NULL OR qm.query_master_id = %s)
        AND (%s::text[] IS NULL OR qm.user_id = ANY(%s))
        """), (start, start, end, end, query_master_id, query_master_id, user_ids, user_ids))
        instance_rows = cur.fetchall()

        #one row per individual result computed within a query_instance, the per-result breakdown
        cur.execute(q("""
        SELECT qr.query_instance_id, rt.name AS result_type_name, qr.start_date, qr.end_date
        FROM {data}.qt_query_result_instance qr
        JOIN {data}.qt_query_result_type rt ON qr.result_type_id = rt.result_type_id
        JOIN {data}.qt_query_instance qi ON qr.query_instance_id = qi.query_instance_id
        JOIN {data}.qt_query_master qm ON qi.query_master_id = qm.query_master_id
        WHERE (%s::timestamp IS NULL OR qm.create_date >= %s)
        AND (%s::timestamp IS NULL OR qm.create_date < %s::timestamp + INTERVAL '1 day')
        AND (%s::int IS NULL OR qm.query_master_id = %s)
        AND (%s::text[] IS NULL OR qm.user_id = ANY(%s))
        """), (start, start, end, end, query_master_id, query_master_id, user_ids, user_ids)) #%s IS NULL makes that half of the OR true, so nothing is filtered
        result_rows = cur.fetchall()


    #one instance can have several results
    results_by_instance = {}
    for query_instance_id, result_type_name, r_start, r_end in result_rows:
        r_duration = (r_end - r_start).total_seconds() if r_end else None
        results_by_instance.setdefault(query_instance_id, []).append({
            "result_type_name": result_type_name,
            "duration": r_duration
        })
       

    result = []
    for query_instance_id, query_master_id, name, user_id, status_name, start_date, end_date, request_xml in instance_rows:
        duration = (end_date - start_date).total_seconds() if end_date else None
        parsed = parse_query_xml(request_xml)
        result.append({
            "query_master_id": query_master_id,
            "name": name,
            "user_id": user_id,
            "status_name": status_name,
            "start_date": start_date,
            "duration": duration,
            "concepts": parsed["item_keys"],
            "included_concepts": parsed["included_concepts"],
            "excluded_concepts": parsed["excluded_concepts"],
            "num_constraints": parsed["num_constraints"],
            "date_constraint_values": parsed["date_constraints"],
            "panels": parsed["panels"],
            "temporal_query_constraints": parsed["temporal_query_constraints"],
            "complexity_score": complexity_score(parsed),
            "results": results_by_instance.get(query_instance_id, [])
        })

    #slowest first, queries with no end_date (still running) sort last, not first
    result.sort(key=lambda r: r["duration"] if r["duration"] is not None else -1, reverse=True)
    return result


def get_user_groups():
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT DISTINCT user_id, value AS group_name
        FROM {pm}.pm_user_params
        WHERE param_name_cd = 'group' AND status_cd = 'A'
        """))
        rows = cur.fetchall()

    result = []
    for user_id, group_name in rows:
        result.append({"user_id": user_id, "group_name": group_name})
    return result


#per month: new registrations, users with at least 1 query, sessions
def get_usage_over_time(start, end, user_ids=None):
    with get_cursor() as cur:
        cur.execute(q("""
        SELECT date_trunc('month', entry_date)::date AS month, COUNT(*)
        FROM {pm}.pm_user_data
        WHERE entry_date >= %s AND entry_date < %s + INTERVAL '1 day'
        AND (%s::text[] IS NULL OR user_id = ANY(%s))
        GROUP BY month
        """), (start, end, user_ids, user_ids))
        new_users = dict(cur.fetchall())

        cur.execute(q("""
        SELECT date_trunc('month', create_date)::date AS month, COUNT(DISTINCT user_id)
        FROM {data}.qt_query_master
        WHERE create_date >= %s AND create_date < %s + INTERVAL '1 day'
        AND request_xml IS NOT NULL
        AND (%s::text[] IS NULL OR user_id = ANY(%s))
        GROUP BY month
        """), (start, end, user_ids, user_ids))
        active_users = dict(cur.fetchall())

        cur.execute(q("""
        SELECT date_trunc('month', entry_date)::date AS month, COUNT(*)
        FROM {pm}.pm_user_session
        WHERE entry_date >= %s AND entry_date < %s + INTERVAL '1 day'
        AND (%s::text[] IS NULL OR user_id = ANY(%s))
        GROUP BY month
        """), (start, end, user_ids, user_ids))
        sessions = dict(cur.fetchall())

    #every month in the range, also the ones without any activity
    result = []
    month = start.replace(day=1)
    while month <= end:
        result.append({
            "month": month.strftime("%Y-%m"),
            "new_users": new_users.get(month, 0),
            "active_users": active_users.get(month, 0),
            "sessions": sessions.get(month, 0),
        })
        month = month.replace(year=month.year + month.month // 12, month=month.month % 12 + 1)
    return result
