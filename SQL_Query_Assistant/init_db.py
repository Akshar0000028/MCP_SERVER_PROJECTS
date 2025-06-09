import sqlite3

conn = sqlite3.connect("data.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    department TEXT
)
""")

cursor.executemany("INSERT INTO employees (name, age, department) VALUES (?, ?, ?)", [
    ("Alice", 30, "HR"),
    ("Bob", 25, "Engineering"),
    ("Charlie", 35, "Marketing")
])

conn.commit()
conn.close()