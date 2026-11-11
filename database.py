import sqlite3
import re
import pandas as pd

DATABASE_PATH = "data/intelligence.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def init_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT UNIQUE,
                row_count INTEGER,
                column_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_dataset_to_database(df, table_name="uploaded_dataset"):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("Invalid SQL table name.")

    init_database()

    with get_connection() as conn:
        df.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False
        )

        conn.execute("""
            INSERT INTO dataset_registry
            (table_name, row_count, column_count)
            VALUES (?, ?, ?)
            ON CONFLICT(table_name)
            DO UPDATE SET
                row_count = excluded.row_count,
                column_count = excluded.column_count,
                created_at = CURRENT_TIMESTAMP
        """, (
            table_name,
            len(df),
            len(df.columns)
        ))

    return table_name


def get_registered_datasets():
    init_database()

    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM dataset_registry ORDER BY id DESC",
            conn
        )


def run_sql_query(query):
    query = query.strip()

    if not query:
        raise ValueError("SQL query cannot be empty.")

    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT and WITH queries are permitted.")

    if ";" in query.rstrip(";"):
        raise ValueError("Only one SQL statement is allowed.")

    init_database()

    with get_connection() as conn:
        conn.execute("PRAGMA query_only = ON")
        return pd.read_sql_query(query, conn)