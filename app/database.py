import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "smartspend.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            email     TEXT UNIQUE NOT NULL,
            budget    REAL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            description TEXT NOT NULL,
            amount      REAL NOT NULL,
            category    TEXT,
            date        DATE DEFAULT CURRENT_DATE,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            category   TEXT NOT NULL,
            limit_amt  REAL NOT NULL,
            month      TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, category, month)
        )
    """)

    # Seed a default user (id=1) so the frontend works without a registration flow
    cursor.execute("""
        INSERT OR IGNORE INTO users (id, username, email, budget)
        VALUES (1, 'User', 'hello@smartspend.ai', 0)
    """)

    conn.commit()
    conn.close()
    print("Database initialized.")