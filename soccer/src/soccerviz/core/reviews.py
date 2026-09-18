"""Append-only analyst annotations, deliberately separate from evaluation labels."""

import sqlite3
from pathlib import Path

import pandas as pd


def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        sample_id TEXT NOT NULL, verdict TEXT NOT NULL, note TEXT NOT NULL)""")
    return connection


def save_review(path: Path, sample_id: str, verdict: str, note: str) -> str:
    if not sample_id or verdict not in {"Correct", "Incorrect", "Uncertain"}:
        raise ValueError("Select a prediction sample and a review verdict")
    with connect(path) as connection:
        connection.execute(
            "INSERT INTO reviews(sample_id, verdict, note) VALUES (?, ?, ?)",
            (sample_id, verdict, note),
        )
    return f"Saved review of {sample_id}. Training labels are unchanged."


def read_reviews(path: Path) -> pd.DataFrame:
    with connect(path) as connection:
        return pd.read_sql_query("SELECT * FROM reviews ORDER BY id DESC", connection)
