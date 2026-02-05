#!/usr/bin/env python3
"""
Recalculate us_t as time since last GLOBAL M4+ earthquake.
No distance constraint - purely time-based.
Uses numpy binary search for speed.
"""

import mysql.connector
import config
import numpy as np
import sys

MAX_MINUTES = 360   # 6 hours cap (model t_max)

def main():
    print("Recalculating us_t = time since last global M4+ earthquake")
    sys.stdout.flush()

    conn = mysql.connector.connect(
        host=config.DB_HOST, user=config.DB_USER,
        password=config.DB_PASS, database=config.DB_NAME,
        autocommit=True
    )
    cursor = conn.cursor()

    # Fetch all records sorted by time
    print("Loading all records...")
    sys.stdout.flush()
    cursor.execute("SELECT us_id, UNIX_TIMESTAMP(us_datetime), us_mag FROM usgs ORDER BY us_datetime ASC")
    rows = cursor.fetchall()
    n = len(rows)
    print(f"Loaded {n:,} records")
    sys.stdout.flush()

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    times = np.array([r[1] for r in rows], dtype=np.float64)
    mags = np.array([r[2] if r[2] is not None else 0.0 for r in rows], dtype=np.float64)

    # Extract M4+ event times (sorted since source query is sorted)
    m4_mask = mags >= 4.0
    m4_times = times[m4_mask]
    print(f"Found {len(m4_times):,} M4+ events")
    sys.stdout.flush()

    # For each record, find time since the most recent M4+ event before it
    print("Calculating dt values (binary search)...")
    sys.stdout.flush()
    new_dt = np.full(n, MAX_MINUTES, dtype=np.int32)

    for i in range(n):
        # Binary search: find the rightmost M4+ event with time < times[i]
        idx = np.searchsorted(m4_times, times[i], side='left') - 1
        if idx >= 0:
            dt_seconds = times[i] - m4_times[idx]
            dt_minutes = max(1, int(dt_seconds / 60))  # min 1 to distinguish from "not calculated"
            new_dt[i] = min(dt_minutes, MAX_MINUTES)
        # else: no M4+ event before this one, keep MAX_MINUTES

        if i % 200000 == 0 and i > 0:
            print(f"Calc: {i:,}/{n:,} ({100*i/n:.0f}%)")
            sys.stdout.flush()

    print(f"Calc: {n:,}/{n:,} (100%)")
    sys.stdout.flush()

    # Stats before DB update
    print(f"\nDistribution preview:")
    for label, lo, hi in [
        ('0-10 min', 0, 10), ('10-30 min', 11, 30), ('30-60 min', 31, 60),
        ('1-2 hours', 61, 120), ('2-4 hours', 121, 240), ('4-6 hours (cap)', 241, 360)
    ]:
        count = np.sum((new_dt >= lo) & (new_dt <= hi))
        pct = 100.0 * count / n
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    sys.stdout.flush()

    # Bulk update via temp table
    print("\nCreating temp table...")
    sys.stdout.flush()
    cursor.execute("DROP TABLE IF EXISTS temp_dt")
    cursor.execute("CREATE TABLE temp_dt (id INT PRIMARY KEY, dt_val INT) ENGINE=InnoDB")

    print("Bulk inserting...")
    sys.stdout.flush()
    chunk_size = 10000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        values_str = ",".join([f"({int(ids[j])},{int(new_dt[j])})" for j in range(start, end)])
        cursor.execute(f"INSERT INTO temp_dt (id, dt_val) VALUES {values_str}")
        if start % 200000 == 0:
            print(f"Insert: {start:,}/{n:,}")
            sys.stdout.flush()

    print(f"Insert: {n:,}/{n:,}")
    print("Bulk UPDATE (single JOIN query)...")
    sys.stdout.flush()
    cursor.execute("UPDATE usgs u JOIN temp_dt t ON u.us_id = t.id SET u.us_t = t.dt_val")
    cursor.execute("DROP TABLE temp_dt")

    print(f"\nDone! Updated {n:,} records")

    # Verify from DB
    print("\nVerification from DB:")
    cursor.execute("""
        SELECT
            CASE
                WHEN us_t <= 10 THEN '0-10 min'
                WHEN us_t <= 30 THEN '10-30 min'
                WHEN us_t <= 60 THEN '30-60 min'
                WHEN us_t <= 120 THEN '1-2 hours'
                WHEN us_t <= 240 THEN '2-4 hours'
                WHEN us_t <= 360 THEN '4-6 hours (cap)'
                ELSE '6h+ (old data)'
            END as time_range,
            COUNT(*) as cnt,
            ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM usgs), 1) as pct
        FROM usgs
        GROUP BY 1
        ORDER BY MIN(us_t)
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,} ({row[2]}%)")

    cursor.close()
    conn.close()
    print("\nReady for training!")

if __name__ == "__main__":
    main()
