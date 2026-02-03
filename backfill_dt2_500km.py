#!/usr/bin/env python3
"""
FAST recalculation of us_t2 with 500km radius.
Uses temp table + single bulk UPDATE for maximum speed.
"""

import mysql.connector
import config
import numpy as np
import sys

RADIUS_KM = 500
MAX_MINUTES = 43200
EARTH_RADIUS_KM = 6371

def haversine_vectorized(lat1, lon1, lats2, lons2):
    lat1_rad = np.radians(lat1)
    lats2_rad = np.radians(lats2)
    dlat = np.radians(lats2 - lat1)
    dlon = np.radians(lons2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lats2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return EARTH_RADIUS_KM * c

def main():
    print(f"FAST us_t2 recalculation with {RADIUS_KM}km radius")
    sys.stdout.flush()

    conn = mysql.connector.connect(
        host=config.DB_HOST, user=config.DB_USER,
        password=config.DB_PASS, database=config.DB_NAME,
        autocommit=True,
        allow_local_infile=True
    )
    cursor = conn.cursor()

    # Fetch all data into numpy
    print("Loading data into memory...")
    sys.stdout.flush()
    cursor.execute("SELECT us_id, us_lat, us_lon, UNIX_TIMESTAMP(us_datetime) FROM usgs ORDER BY us_datetime ASC")
    data = np.array(cursor.fetchall(), dtype=np.float64)
    ids = data[:, 0].astype(np.int64)
    lats = data[:, 1]
    lons = data[:, 2]
    times = data[:, 3]
    n = len(ids)
    print(f"Loaded {n:,} records")
    sys.stdout.flush()

    # Calculate all values in memory
    print("Calculating dt2 values (vectorized)...")
    sys.stdout.flush()
    new_t2 = np.full(n, MAX_MINUTES, dtype=np.int32)
    lat_box = RADIUS_KM / 111.0
    max_seconds = MAX_MINUTES * 60
    window_start = 0

    for i in range(1, n):
        curr_time, curr_lat, curr_lon = times[i], lats[i], lons[i]

        while window_start < i and (curr_time - times[window_start]) > max_seconds:
            window_start += 1
        if window_start >= i:
            continue

        cand_lats = lats[window_start:i]
        cand_lons = lons[window_start:i]
        cand_times = times[window_start:i]

        lat_mask = np.abs(cand_lats - curr_lat) <= lat_box
        if not np.any(lat_mask):
            continue

        f_lats = cand_lats[lat_mask]
        f_lons = cand_lons[lat_mask]
        f_times = cand_times[lat_mask]

        lon_box = RADIUS_KM / (111.0 * max(0.1, np.cos(np.radians(abs(curr_lat)))))
        lon_mask = np.abs(f_lons - curr_lon) <= lon_box
        if not np.any(lon_mask):
            continue

        f2_lats = f_lats[lon_mask]
        f2_lons = f_lons[lon_mask]
        f2_times = f_times[lon_mask]

        distances = haversine_vectorized(curr_lat, curr_lon, f2_lats, f2_lons)
        within = distances <= RADIUS_KM
        if np.any(within):
            most_recent = np.max(f2_times[within])
            new_t2[i] = min(int((curr_time - most_recent) / 60), MAX_MINUTES)

        if i % 100000 == 0:
            print(f"Calc: {i:,}/{n:,} ({100*i/n:.0f}%)")
            sys.stdout.flush()

    print(f"Calc: {n:,}/{n:,} (100%)")
    print("Creating temp table...")
    sys.stdout.flush()

    # Create temp table
    cursor.execute("DROP TABLE IF EXISTS temp_t2")
    cursor.execute("CREATE TABLE temp_t2 (id INT PRIMARY KEY, t2 INT) ENGINE=InnoDB")

    print("Bulk inserting to temp table...")
    sys.stdout.flush()

    # Build and execute large INSERT statements
    chunk_size = 10000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Build VALUES clause directly (faster than executemany)
        values_str = ",".join([f"({int(ids[j])},{int(new_t2[j])})" for j in range(start, end)])
        cursor.execute(f"INSERT INTO temp_t2 (id, t2) VALUES {values_str}")
        if start % 200000 == 0:
            print(f"Insert: {start:,}/{n:,}")
            sys.stdout.flush()

    print(f"Insert: {n:,}/{n:,}")
    print("Bulk UPDATE (single JOIN query)...")
    sys.stdout.flush()
    cursor.execute("UPDATE usgs u JOIN temp_t2 t ON u.us_id = t.id SET u.us_t2 = t.t2")
    cursor.execute("DROP TABLE temp_t2")

    print(f"\nDone! Updated {n:,} records")

    # Show distribution
    print("\nNew us_t2 distribution:")
    cursor.execute("""
        SELECT
            CASE
                WHEN us_t2 <= 60 THEN '0-60 min'
                WHEN us_t2 <= 360 THEN '1-6 hours'
                WHEN us_t2 <= 720 THEN '6-12 hours'
                WHEN us_t2 <= 1440 THEN '12-24 hours'
                WHEN us_t2 <= 10080 THEN '1-7 days'
                WHEN us_t2 <= 43200 THEN '7-30 days'
                ELSE '30+ days'
            END as time_range,
            COUNT(*) as cnt,
            ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM usgs), 1) as pct
        FROM usgs
        GROUP BY 1
        ORDER BY MIN(us_t2)
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,} ({row[2]}%)")

    cursor.close()
    conn.close()
    print("\nReady for training!")

if __name__ == "__main__":
    main()
