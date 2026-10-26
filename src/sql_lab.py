"""Read-only SQL access to the analytical database, for the SQL Lab and the AI Copilot.

Three independent guards keep user- or LLM-written SQL harmless:
1. the database is opened read-only (SQLite `mode=ro` URI);
2. an authorizer rejects everything except reading (no writes, ATTACH, PRAGMA, ...);
3. a progress handler aborts runaway queries, and results are capped in rows.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pandas as pd

from src import configs as config
from src.data_preprocessing import connect, named_queries

MAX_ROWS = 5000
TIME_LIMIT_S = 5.0

_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
            getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


class QueryError(ValueError):
    """The query was rejected or failed; the message is safe to show to the user."""


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        with closing(connect(db_path)):      # builds the database on a fresh clone
            pass
    con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True,
                          check_same_thread=False)
    con.set_authorizer(_authorizer)
    return con


def run_query(sql: str, db_path: Path = config.DB_PATH,
              max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, bool]:
    """Run one read-only statement. Returns (rows, truncated?)."""
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        raise QueryError("The query is empty.")
    with closing(_readonly_connection(db_path)) as con:
        deadline = time.monotonic() + TIME_LIMIT_S
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = con.execute(sql)
            columns = [c[0] for c in cur.description or []]
            rows = cur.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as e:
            msg = str(e)
            if "not authorized" in msg:
                msg = "Only read-only SELECT queries are allowed."
            elif "interrupted" in msg:
                msg = f"The query took longer than {TIME_LIMIT_S:.0f} seconds and was stopped."
            raise QueryError(msg) from None
        except sqlite3.ProgrammingError as e:  # e.g. more than one statement
            raise QueryError(str(e)) from None
    if not columns:
        raise QueryError("The statement returned no result set.")
    truncated = len(rows) > max_rows
    return pd.DataFrame(rows[:max_rows], columns=columns), truncated


def schema(db_path: Path = config.DB_PATH) -> dict[str, list[tuple[str, str]]]:
    """{table_or_view: [(column, type), ...]} for the user-facing tables."""
    out = {}
    with closing(connect(db_path)) as con:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' AND name <> 'raw_customers' ORDER BY type, name")]
        for name in names:
            out[name] = [(r[1], r[2] or "") for r in con.execute(f"PRAGMA table_info('{name}')")]
    return out


def schema_text(db_path: Path = config.DB_PATH) -> str:
    return "\n".join(f"{t}({', '.join(f'{c} {ty}'.strip() for c, ty in cols)})"
                     for t, cols in schema(db_path).items())


def templates() -> dict[str, str]:
    """Ready-made queries: every named BI query plus a few exploratory ones."""
    target = config.TARGET
    extra = {
        "top_20_customers_by_expected_loss": (
            "SELECT s.customer_id, ROUND(s.churn_probability, 3) AS p_churn, s.risk_level,\n"
            "       ROUND(s.expected_loss, 2) AS expected_loss, c.total_spend,\n"
            "       c.years_as_customer, c.last_purchase_days_ago, c.satisfaction_score,\n"
            "       s.top_reasons\n"
            "FROM customer_scores s JOIN customers c USING (customer_id)\n"
            f"WHERE c.{target} = 0\nORDER BY s.expected_loss DESC\nLIMIT 20"),
        "risk_by_promotion_response": (
            "SELECT c.promotion_response, COUNT(*) AS active_customers,\n"
            "       ROUND(AVG(s.churn_probability) * 100, 1) AS avg_risk_pct,\n"
            "       ROUND(SUM(s.expected_loss), 0) AS expected_loss\n"
            "FROM customer_scores s JOIN customers c USING (customer_id)\n"
            f"WHERE c.{target} = 0\nGROUP BY c.promotion_response\nORDER BY expected_loss DESC"),
        "expected_loss_by_segment": (
            "SELECT s.segment, COUNT(*) AS active_customers,\n"
            "       ROUND(AVG(s.churn_probability) * 100, 1) AS avg_risk_pct,\n"
            "       ROUND(SUM(s.expected_loss), 0) AS expected_loss\n"
            "FROM customer_scores s JOIN customers c USING (customer_id)\n"
            f"WHERE c.{target} = 0\nGROUP BY s.segment\nORDER BY expected_loss DESC"),
        "lapsed_high_risk_customers": (
            "SELECT s.customer_id, ROUND(s.churn_probability, 3) AS p_churn,\n"
            "       c.last_purchase_days_ago, c.num_of_purchases,\n"
            "       ROUND(s.expected_loss, 2) AS expected_loss, s.recommended_action\n"
            "FROM customer_scores s JOIN customers c USING (customer_id)\n"
            f"WHERE c.{target} = 0 AND s.risk_level = 'HIGH' AND c.last_purchase_days_ago >= 90\n"
            "ORDER BY s.expected_loss DESC\nLIMIT 50"),
    }
    return {**extra, **named_queries()}