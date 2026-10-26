from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import pandas as pd
from src import configs

SCHEMA_SQL = configs.SQL_DIR / "01_create_schema.sql"
QUALITY_SQL = configs.SQL_DIR / "02_data_quality_checks.sql"
QUERIES_SQL = configs.SQL_DIR / "03_business_queries.sql"

class DataQualityError(RuntimeError):
    """Raised when a data-quality check reports violations."""


#-------------------------------------------
# Loading
#-------------------------------------------

def load_raw_csv(path: Path = configs.RAW_CSV) -> pd.DataFrame:
    """Read the source file exactly as delivered: every column as text.

    Typing happens in SQL, where a bad cast is visible, instead of letting
    pandas silently turn " " into NaN or "0012" into 12.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def _land_raw(con: sqlite3.Connection, raw: pd.DataFrame) -> None:
    missing = [c for c in configs.RAW_REQUIRED_COLUMNS if c not in raw.columns]

    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    raw = raw.copy()
    if "target_churn" not in raw.columns.str.lower(): # scoring files have no label yet
        raw['target_churn'] = ""
    raw.astype(str).to_sql('raw_customers', con, if_exists='replace', index = False)
    con.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))

def build_database(raw: pd.DataFrame | None = None, db_path:Path = configs.DB_PATH,
                   check_quality: bool = True) -> Path:
    """(Re)build the SQLite analytical layer from the raw CSV.

    If the pipeline has already scored the customers, the 'customer_scores'
    table is resotred too, so a rebuild (or a fresh clone, where the
    database is not comitted) never leaves the BI queries without their scores.
    """
    raw = load_raw_csv() if raw is None else raw
    db_path.unlink(missing_ok = True)
    with closing(sqlite3.connect(db_path)) as con:
        _land_raw(con, raw)
        con.commit()
        if check_quality:
            report = run_quality_checks(con)
            failed = report[report['violations'] > 0]
            if not failed.empty:
                raise DataQualityError(
                    "Data-qualty checks failed:\n" + failed.to_string(index=False))

    if configs.SCORED_CSV.exists():
        write_scores(pd.read_csv(configs.SCORED_CSV), db_path)
    return db_path

def connect(db_path: Path = configs.DB_PATH) -> sqlite3.Connection:
    """Open the analytical database, building it first on a fresh clone."""
    if not db_path.exists():
        build_database(db_path=db_path)
    return sqlite3.connect(db_path, check_same_thread=False)

#------------------------------------------------------------------------------
# Quality Checks
#------------------------------------------------------------------------------
def _split_statements(sql: str) -> list[str]:
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]

def run_quality_checks(con: sqlite3.Connection) -> pd.DataFrame:
    """Run every check in 02_data_quality_checks.sql; one row per check."""
    statements = _split_statements(QUALITY_SQL.read_text(encoding= "utf-8"))
    return pd.concat([pd.read_sql_query(s, con) for s in statements], ignore_index=True)

#--------------------------------------
# Named Business Queries
#--------------------------------------
@lru_cache(maxsize=1)
def named_queries() -> dict[str, str]:
    """Parse 03_business_quieries.sql into {name: sql}."""
    text = QUERIES_SQL.read_text(encoding='utf-8')
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    return {name: sql.strip().rstrip(";") for name , sql in zip(parts[1::2], parts[2::2])}

def run_named_query(name: str, con: sqlite3.Connection | None = None) -> pd.DataFrame:
    sql = named_queries()[name]
    if con is not None:
        return pd.read_sql_query(sql, con)
    with closing(connect()) as own:
        return pd.read_sql_query(sql, own)

#--------------------------------------
# Feature tables
#--------------------------------------
def load_features(con: sqlite3.Connection | None = None) -> pd.DataFrame:
    """The modelling table: Cleaned Customers + SQL derived business features."""
    if con is not None:
        return pd.read_sql_query("SELECT * FROM customer_features", con)
    with closing(connect()) as own:
        return pd.read_sql_query("SELECT * FROM customer_features", own)

def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and feature-engineer an arbitrary raw customer file.
    
    Runs the exact SQL used for tarining against a throwaway in-memory
    database, so uploadedfiles are treated identically to training data.
    """
    has_label = "target_churn" in raw.columns and raw["target_churn"].astype(str).str.strip().ne("").any()
    with closing(sqlite3.connect(":memory:")) as con:
        _land_raw(con, raw.astype(str))
        features = pd.read_sql_query("SELECT * FROM customer_features", con)
    if not has_label:
        features = features.drop(columns=[configs.TARGET])
    return features

def write_table(df: pd.DataFrame, table:str, db_path: Path = configs.DB_PATH) -> None:
    with closing(sqlite3.connect(db_path)) as con:
        df.to_sql(table, con, if_exists="replace", index=False)

def write_scores(scored:pd.DataFrame, db_path:Path = configs.DB_PATH) -> None:
    """Publish model scores to SQL as 'customer_scores' (used by 03_business_queries).
    
    Indexed on customer_id: every BI query joins it to 'customers' and without index SQLite 
    falls back to a nested loop scan of both tables.
    """
    write_table(scored[configs.SCORE_TABLE_COLUMNS], "customer_scores", db_path)
    with closing(sqlite3.connect(db_path)) as con:
        con.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_scores_id
                    ON customer_scores (customer_id)""")

        con.commit()

if __name__ == "__main__":
    path = build_database()
    with closing(sqlite3.connect(path)) as con:
        print(run_quality_checks(con).to_string(index=False))
        feats = load_features(con)
    feats.to_csv(configs.FEATURES_CSV,index = False)
    print("customer_features.csv saved")