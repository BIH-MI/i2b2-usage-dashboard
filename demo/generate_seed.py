#generates demo/seed.sql (made-up i2b2 tables and data) and demo/users.json (demo login)
#only the tables and columns the dashboard reads, all data is invented, no real users or queries.
#times are written relative to now(), so the demo always shows recent activity
#run from the project root: python demo/generate_seed.py
import json
import random
from pathlib import Path
from xml.sax.saxutils import escape

import bcrypt

OUT_DIR = Path(__file__).parent
DAYS = 540 #about 18 months of history
DEMO_USER = "demo"
DEMO_PASSWORD = "demo1234"

rng = random.Random(42) #fixed seed, same data on every run

#i2b2 style ontology paths, the last segment is what the dashboard shows
CONCEPTS = {
    "Diagnoses": [
        "Diabetes mellitus type 2 (E11)", "Essential hypertension (I10)", "Heart failure (I50)",
        "Asthma (J45)", "COPD (J44)", "Chronic kidney disease (N18)", "Atrial fibrillation (I48)",
        "Depressive episode (F32)", "Breast cancer (C50)", "Stroke (I63)", "Obesity (E66)",
        "Rheumatoid arthritis (M06)",
    ],
    "Laboratory": [
        "HbA1c", "Creatinine", "eGFR", "LDL cholesterol", "Hemoglobin", "CRP", "TSH", "NT-proBNP",
    ],
    "Medications": [
        "Metformin", "Insulin", "ACE inhibitors", "Beta blockers", "Statins", "Anticoagulants",
        "Corticosteroids",
    ],
    "Procedures": [
        "Coronary angiography", "Dialysis", "Echocardiography", "Colonoscopy", "Knee replacement",
    ],
    "Demographics": ["Female", "Male", "Age 18-39", "Age 40-64", "Age 65+"],
}
LAB_UNITS = {
    "HbA1c": ("%", 5, 12), "Creatinine": ("mg/dL", 0.5, 4), "eGFR": ("mL/min", 15, 120),
    "LDL cholesterol": ("mg/dL", 70, 220), "Hemoglobin": ("g/dL", 8, 17), "CRP": ("mg/L", 1, 100),
    "TSH": ("mU/L", 0.3, 8), "NT-proBNP": ("pg/mL", 100, 5000),
}
#diagnoses and labs are queried most
CATEGORY_WEIGHTS = {"Diagnoses": 5, "Laboratory": 3, "Medications": 2, "Procedures": 1, "Demographics": 2}

STATUS_TYPES = [(1, "QUEUED"), (2, "PROCESSING"), (3, "FINISHED"), (4, "ERROR"), (5, "INCOMPLETE"),
                (6, "COMPLETED"), (7, "MEDIUM_QUEUE"), (8, "LARGE_QUEUE"), (9, "CANCELLED"), (10, "TIMEDOUT")]
STATUS_WEIGHTS = {3: 86, 4: 5, 5: 2, 9: 3, 10: 2}
RESULT_TYPES = [(1, "PATIENTSET"), (2, "PATIENT_COUNT_XML"), (3, "PATIENT_GENDER_COUNT_XML"),
                (4, "PATIENT_AGE_COUNT_XML")]

GROUPS = ["research", "clinic", "data-office"]


def ago(minutes):
    return f"now() - interval '{int(minutes)} minutes'"


def concept_path(category, name):
    return f"\\\\i2b2_DEMO\\i2b2\\{category}\\{name}\\"


def pick_concept():
    category = rng.choices(list(CATEGORY_WEIGHTS), weights=CATEGORY_WEIGHTS.values())[0]
    return category, rng.choice(CONCEPTS[category])


def item_xml(category, name):
    parts = [f"<item><item_key>{escape(concept_path(category, name))}</item_key><item_name>{escape(name)}</item_name>"]
    if category == "Laboratory" and rng.random() < 0.4:
        unit, low, high = LAB_UNITS[name]
        value = round(rng.uniform(low, high), 1)
        operator = rng.choice(["GT", "LT", "GE"])
        parts.append(f"<constrain_by_value><value_operator>{operator}</value_operator>"
                     f"<value_constraint>{value}</value_constraint><value_unit_of_measure>{unit}</value_unit_of_measure>"
                     f"<value_type>NUMBER</value_type></constrain_by_value>")
    if rng.random() < 0.15:
        year = rng.randint(2012, 2025)
        month = rng.randint(1, 12)
        length = rng.randint(3, 36)
        end_month = month + length
        end_year = year + (end_month - 1) // 12
        end_month = (end_month - 1) % 12 + 1
        parts.append(f"<constrain_by_date><date_from>{year}-{month:02d}-01T00:00:00.000+01:00</date_from>"
                     f"<date_to>{end_year}-{end_month:02d}-01T00:00:00.000+01:00</date_to></constrain_by_date>")
    parts.append("</item>")
    return "".join(parts)


def panel_xml(number, exclude):
    occurrences = rng.choice([2, 3, 5]) if rng.random() < 0.1 else 1
    concepts = {pick_concept() for _ in range(rng.choices([1, 2, 3], weights=[6, 3, 1])[0])} #OR inside a panel
    items = "".join(item_xml(c, n) for c, n in concepts)
    return (f"<panel><panel_number>{number}</panel_number><invert>{1 if exclude else 0}</invert>"
            f"<panel_timing>ANY</panel_timing><total_item_occurrences>{occurrences}</total_item_occurrences>{items}</panel>")


#returns (name, request_xml), panels are AND, items in a panel are OR
def query_definition():
    panel_count = rng.choices([1, 2, 3, 4, 5], weights=[3, 4, 3, 2, 1])[0]
    panels = [panel_xml(i + 1, exclude=(i > 0 and rng.random() < 0.18)) for i in range(panel_count)]
    temporal = ""
    if panel_count >= 2 and rng.random() < 0.1: #event 1 happens before event 2
        temporal = ("<subquery_constraint><first_query><query_id>Event 1</query_id></first_query>"
                    "<operator>LESS</operator><second_query><query_id>Event 2</query_id></second_query>"
                    f"<span><operator>LESSEQUAL</operator><span_value>{rng.choice([30, 90, 365])}</span_value>"
                    "<units>DAY</units></span></subquery_constraint>")
    first_item = panels[0].split("<item_name>")[1].split("</item_name>")[0]
    name = first_item.split(" (")[0][:30]
    xml = (f"<query_definition><query_name>{escape(name)}</query_name><specificity_scale>0</specificity_scale>"
           f"{''.join(panels)}{temporal}</query_definition>")
    return name, xml


def sql_value(v):
    if v is None:
        return "NULL"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str) and v.startswith("now()"):
        return v
    return "'" + str(v).replace("'", "''") + "'"


def insert(table, columns, rows):
    if not rows:
        return ""
    out = []
    for i in range(0, len(rows), 500):
        values = ",\n".join("(" + ", ".join(sql_value(v) for v in row) + ")" for row in rows[i:i + 500])
        out.append(f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n{values};\n")
    return "".join(out)


def main():
    users = []
    for i in range(1, 25):
        joined = rng.uniform(30, DAYS * 0.98) if i > 6 else DAYS * rng.uniform(0.97, 1.0) #6 early users
        users.append({
            "id": f"user{i:02d}",
            "joined": joined * 24 * 60, #minutes ago
            "activity": rng.choice([0.2, 0.5, 1, 1, 2, 4]), #sessions per week
            "skill": rng.random(), #more panels and constraints for higher skill
            "group": rng.choice(GROUPS),
        })

    masters, instances, results, sessions, logins, locked = [], [], [], [], [], []
    master_id = instance_id = result_id = 0
    session_id = 0
    saved = [] #earlier definitions, sometimes run again (i2b2 makes a new master for every run)

    for u in users:
        minute = u["joined"]
        while minute > 60:
            minute -= rng.expovariate(u["activity"] / (7 * 24 * 60))
            if minute <= 60:
                break
            length = rng.uniform(20, 180)
            session_id += 1
            sid = f"demo-session-{session_id}"
            sessions.append((u["id"], sid, ago(minute), ago(max(minute - length, 0))))
            logins.append((u["id"], "SUCCESS", ago(minute + 0.5)))
            if rng.random() < 0.05:
                logins.append((u["id"], "BADPASSWORD", ago(minute + 2)))

            for _ in range(rng.choices([0, 1, 2, 3, 5], weights=[2, 4, 3, 2, 1])[0]):
                created = minute - rng.uniform(1, length)
                if saved and rng.random() < 0.12:
                    name, xml = rng.choice(saved)
                else:
                    name, xml = query_definition()
                    if u["skill"] < 0.3 and xml.count("<panel>") > 2: #beginners build smaller queries
                        name, xml = query_definition()
                    saved.append((name, xml))
                master_id += 1
                clock = f"{rng.randint(7, 19):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
                masters.append((master_id, f"{name}@{clock}", u["id"], ago(created), xml))

                status = rng.choices(list(STATUS_WEIGHTS), weights=STATUS_WEIGHTS.values())[0]
                complexity = xml.count("<item>") + 3 * xml.count("<constrain_") + 5 * xml.count("<subquery_constraint>")
                seconds = rng.lognormvariate(1.5 + 0.12 * complexity, 0.9)
                if status == 10:
                    seconds = rng.uniform(600, 1800)
                start = created - 0.05
                end = ago(max(start - seconds / 60, 0))
                instance_id += 1
                instances.append((instance_id, master_id, ago(start), end, status))

                if status in (3, 6):
                    picked = [1, 2] + rng.sample([3, 4], rng.choice([0, 0, 1, 2]))
                    for result_type in picked:
                        result_id += 1
                        part = seconds * rng.uniform(0.2, 1.0)
                        results.append((result_id, instance_id, result_type, ago(start), ago(max(start - part / 60, 0))))

        if rng.random() < 0.1:
            logins.append(("unknown.user", "NONEXIST", ago(rng.uniform(60, DAYS * 24 * 60))))
    locked.append(("user07", "LOCKEDOUT", ago(rng.uniform(60, 20 * 24 * 60))))

    sql = ["-- generated by demo/generate_seed.py, invented data for the dashboard demo only\n",
           "CREATE SCHEMA i2b2demodata;\nCREATE SCHEMA i2b2pm;\n\n",
           """CREATE TABLE i2b2demodata.qt_query_status_type (status_type_id int PRIMARY KEY, name varchar(100));
CREATE TABLE i2b2demodata.qt_query_result_type (result_type_id int PRIMARY KEY, name varchar(100));
CREATE TABLE i2b2demodata.qt_query_master (query_master_id int PRIMARY KEY, name varchar(250), user_id varchar(50),
    create_date timestamp, request_xml text);
CREATE TABLE i2b2demodata.qt_query_instance (query_instance_id int PRIMARY KEY, query_master_id int,
    start_date timestamp, end_date timestamp, status_type_id int);
CREATE TABLE i2b2demodata.qt_query_result_instance (result_instance_id int PRIMARY KEY, query_instance_id int,
    result_type_id int, start_date timestamp, end_date timestamp);
CREATE TABLE i2b2pm.pm_user_data (user_id varchar(50) PRIMARY KEY, entry_date timestamp);
CREATE TABLE i2b2pm.pm_user_session (user_id varchar(50), session_id varchar(50), entry_date timestamp, expired_date timestamp);
CREATE TABLE i2b2pm.pm_user_login (user_id varchar(50), attempt_cd varchar(50), entry_date timestamp);
CREATE TABLE i2b2pm.pm_user_params (user_id varchar(50), param_name_cd varchar(50), value text, status_cd varchar(50));
CREATE TABLE i2b2pm.pm_project_user_params (user_id varchar(50), param_name_cd varchar(50), entry_date timestamp);

""",
           insert("i2b2demodata.qt_query_status_type", ["status_type_id", "name"], STATUS_TYPES),
           insert("i2b2demodata.qt_query_result_type", ["result_type_id", "name"], RESULT_TYPES),
           insert("i2b2pm.pm_user_data", ["user_id", "entry_date"], [(u["id"], ago(u["joined"])) for u in users]),
           insert("i2b2pm.pm_user_params", ["user_id", "param_name_cd", "value", "status_cd"],
                  [(u["id"], "group", u["group"], "A") for u in users]),
           insert("i2b2pm.pm_project_user_params", ["user_id", "param_name_cd", "entry_date"], locked),
           insert("i2b2pm.pm_user_session", ["user_id", "session_id", "entry_date", "expired_date"], sessions),
           insert("i2b2pm.pm_user_login", ["user_id", "attempt_cd", "entry_date"], logins),
           insert("i2b2demodata.qt_query_master", ["query_master_id", "name", "user_id", "create_date", "request_xml"], masters),
           insert("i2b2demodata.qt_query_instance", ["query_instance_id", "query_master_id", "start_date", "end_date", "status_type_id"], instances),
           insert("i2b2demodata.qt_query_result_instance", ["result_instance_id", "query_instance_id", "result_type_id", "start_date", "end_date"], results)]

    (OUT_DIR / "seed.sql").write_text("".join(sql), encoding="utf-8")

    hashed = bcrypt.hashpw(DEMO_PASSWORD.encode(), bcrypt.gensalt()).decode()
    (OUT_DIR / "users.json").write_text(json.dumps({DEMO_USER: {"hash": hashed, "role": "admin"}}, indent=2), encoding="utf-8")

    print(f"{len(users)} users, {len(sessions)} sessions, {len(masters)} queries, {len(results)} results")


if __name__ == "__main__":
    main()
