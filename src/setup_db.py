import sqlite3

conn = sqlite3.connect("data/sample.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    product TEXT,
    region TEXT,
    amount REAL
)
""")

cur.executemany(
    "INSERT INTO sales (product, region, amount) VALUES (?, ?, ?)",
    [
        ("Widget A", "North", 1200.50),
        ("Widget B", "South", 850.00),
        ("Widget A", "East", 430.75),
        ("Widget C", "West", 2100.00),
        ("Widget B", "North", 990.25),
    ],
)

conn.commit()
conn.close()
print("Database created at data/sample.db")