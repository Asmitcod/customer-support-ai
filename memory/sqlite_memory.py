import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


class SQLiteMemory:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_name TEXT NOT NULL,
                    customer_query TEXT NOT NULL,
                    detected_department TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save_interaction(
        self,
        customer_name: str,
        customer_query: str,
        detected_department: str,
        assistant_response: str,
    ) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    customer_name,
                    customer_query,
                    detected_department,
                    assistant_response,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    customer_name or "Unknown",
                    customer_query,
                    detected_department,
                    assistant_response,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            connection.commit()

    def get_previous_issue(self, customer_name: str = "Unknown") -> Optional[str]:
        query = """
            SELECT customer_query, detected_department, timestamp
            FROM interactions
            WHERE detected_department != 'Memory Recall'
        """
        parameters: tuple[str, ...] = ()

        if customer_name and customer_name != "Unknown":
            query += " AND customer_name = ?"
            parameters = (customer_name,)

        query += " ORDER BY id DESC LIMIT 1"

        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(query, parameters).fetchone()

        if not row:
            return None

        customer_query, department, timestamp = row
        return f"Previous issue: {customer_query} | Department: {department} | Stored at: {timestamp}"
