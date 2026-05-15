import sqlite3
import json
from datetime import datetime

class Database:
    def __init__(self, db_path="prompt_optimizer.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            status TEXT,
            current_step TEXT
        )
        """)
        
        # Workflow steps table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS workflow_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            step_name TEXT,
            input_data TEXT,
            output_data TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """)
        
        # Memory table (for context)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            key TEXT,
            value TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """)
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (session_id, datetime.now(), "initialized", "input_processing")
        )
        conn.commit()
        conn.close()

    def update_session(self, session_id: str, status: str, current_step: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET status = ?, current_step = ? WHERE id = ?",
            (status, current_step, session_id)
        )
        conn.commit()
        conn.close()
    
    def log_step(self, session_id: str, step_name: str, input_data: dict, output_data: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO workflow_steps (session_id, step_name, input_data, output_data, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, step_name, json.dumps(input_data), json.dumps(output_data), datetime.now())
        )
        conn.commit()
        conn.close()

    def upsert_memory(self, session_id: str, key: str, value: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, value FROM memory WHERE session_id = ? AND key = ? ORDER BY id DESC LIMIT 1",
            (session_id, key)
        )
        existing = cursor.fetchone()

        if existing:
            current = json.loads(existing[1])
            merged = {**current, **value} if isinstance(current, dict) else value
            cursor.execute(
                "UPDATE memory SET value = ?, created_at = ? WHERE id = ?",
                (json.dumps(merged), datetime.now(), existing[0])
            )
        else:
            cursor.execute(
                "INSERT INTO memory (session_id, key, value, created_at) VALUES (?, ?, ?, ?)",
                (session_id, key, json.dumps(value), datetime.now())
            )

        conn.commit()
        conn.close()

    def get_memory(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM memory WHERE session_id = ?", (session_id,))
        results = cursor.fetchall()
        conn.close()
        return {key: json.loads(value) for key, value in results}
    
    def get_session_history(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT step_name, input_data, output_data, timestamp FROM workflow_steps WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "step": r[0],
                "input": json.loads(r[1]),
                "output": json.loads(r[2]),
                "timestamp": r[3]
            }
            for r in results
        ]
