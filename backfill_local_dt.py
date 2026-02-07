#!/usr/bin/env python3
"""
Backfill us_lt = minutes since last M4+ earthquake within 1000km (haversine).

FULLY VECTORIZED — no Python per-record loops.
Algorithm: iteratively check the next-most-recent M4+ for ALL records simultaneously.
~95% of records resolve on iteration 1, remaining shrink exponentially.
"""

import mysql.connector
import config
import numpy as np
import sys
import time as time_mod

MAX_MINUTES = 29000000  # ~55 years sentinel
RADIUS_KM = 1000.0
EARTH_R = 6371.0
MAX_ITER = 10000  # safety limit for backward scan


def main():
    t0 = time_mod.time()
    print("Backfilling us_lt = time since last M4+ within 1000km (VECTORIZED)")
    sys.stdout.flush()

    conn = mysql.connector.connect(
        host=config.DB_HOST, user=config.DB_USER,
        password=config.DB_PASS, database=config.DB_NAME,
        autocommit=True, ssl_disabled=True
    )
    cursor = conn.cursor()

    # ==================== LOAD DATA ====================
    print("Loading all records...")
    sys.stdout.flush()
    cursor.execute(
        "SELECT us_id, UNIX_TIMESTAMP(us_datetime), us_lat, us_lon, us_mag "
        "FROM usgs ORDER BY us_datetime ASC"
    )
    rows = cursor.fetchall()
    n = len(rows)
    t1 = time_mod.time()
    print(f"Loaded {n:,} records ({t1-t0:.1f}s)")
    sys.stdout.flush()

    ids   = np.array([r[0] for r in rows], dtype=np.int64)
    times = np.array([r[1] for r in rows], dtype=np.float64)
    lats  = np.array([r[2] if r[2] is not None else 0.0 for r in rows], dtype=np.float64)
    lons  = np.array([r[3] if r[3] is not None else 0.0 for r in rows], dtype=np.float64)
    mags  = np.array([r[4] if r[4] is not None else 0.0 for r in rows], dtype=np.float64)
    del rows  # free ~200MB

    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)

    # Build sorted M4+ arrays
    m4_mask = mags >= 4.0
    m4_times    = times[m4_mask]
    m4_lats_rad = lats_rad[m4_mask]
    m4_lons_rad = lons_rad[m4_mask]
    n_m4 = len(m4_times)
    print(f"Found {n_m4:,} M4+ events")
    sys.stdout.flush()

    # ==================== VECTORIZED COMPUTATION ====================
    print("Computing local dt (fully vectorized)...")
    sys.stdout.flush()
    tc = time_mod.time()

    result = np.full(n, MAX_MINUTES, dtype=np.int64)
    found = np.zeros(n, dtype=bool)

    # For each record, find the last M4+ with time STRICTLY BEFORE it
    # side='left' → first index where m4_time >= record_time, minus 1 = last m4 < record_time
    current_m4_idx = np.searchsorted(m4_times, times, side='left').astype(np.int64) - 1

    for iteration in range(MAX_ITER):
        # Active = not yet found AND still has M4+ candidates to check
        active = ~found & (current_m4_idx >= 0)
        n_active = active.sum()
        if n_active == 0:
            break

        aidx = np.where(active)[0]       # indices into full arrays
        m4i = current_m4_idx[aidx]        # M4+ indices to check

        # Vectorized haversine for ALL active records at once
        dlat = m4_lats_rad[m4i] - lats_rad[aidx]
        dlon = m4_lons_rad[m4i] - lons_rad[aidx]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lats_rad[aidx]) * np.cos(m4_lats_rad[m4i]) * np.sin(dlon / 2.0) ** 2
        dists = 2.0 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        within = dists <= RADIUS_KM

        # Records that found a match
        found_idx = aidx[within]
        found_m4 = m4i[within]
        dt_sec = times[found_idx] - m4_times[found_m4]
        result[found_idx] = np.maximum(1, (dt_sec / 60.0).astype(np.int64))
        found[found_idx] = True

        # Records that didn't — move to previous M4+ event
        not_found_idx = aidx[~within]
        current_m4_idx[not_found_idx] -= 1

        if iteration % 50 == 0:
            elapsed = time_mod.time() - tc
            print(f"  iter {iteration:5d} | found: {found.sum():>10,}/{n:,} ({100*found.sum()/n:.1f}%) | active: {n_active:>10,} | {elapsed:.1f}s")
            sys.stdout.flush()

    td = time_mod.time()
    n_found = found.sum()
    n_none = n - n_found
    print(f"\nComputation done in {td-tc:.1f}s ({iteration+1} iterations)")
    print(f"  Found local M4+: {n_found:,} ({100*n_found/n:.1f}%)")
    print(f"  No local M4+:    {n_none:,} ({100*n_none/n:.1f}%) → sentinel {MAX_MINUTES}")
    sys.stdout.flush()

    # ==================== DISTRIBUTION ====================
    print(f"\nDistribution preview:")
    for label, lo, hi in [
        ('0-10 min',          1,       10),
        ('10-60 min',        11,       60),
        ('1-6 hours',        61,      360),
        ('6-24 hours',      361,     1440),
        ('1-7 days',       1441,    10080),
        ('7-30 days',     10081,    43200),
        ('1-12 months',   43201,   525600),
        ('1-10 years',   525601,  5256000),
        ('10+ years',   5256001, MAX_MINUTES - 1),
        ('no local M4+', MAX_MINUTES, MAX_MINUTES),
    ]:
        count = int(np.sum((result >= lo) & (result <= hi)))
        pct = 100.0 * count / n
        print(f"  {label:15s}: {count:>10,} ({pct:5.1f}%)")
    sys.stdout.flush()

    # ==================== BULK UPDATE ====================
    print("\nCreating temp table...")
    sys.stdout.flush()
    cursor.execute("DROP TABLE IF EXISTS temp_local_dt")
    cursor.execute("CREATE TABLE temp_local_dt (id INT PRIMARY KEY, dt_val INT) ENGINE=InnoDB")

    print("Bulk inserting...")
    sys.stdout.flush()
    chunk_size = 50000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        values_str = ",".join(
            [f"({int(ids[j])},{int(result[j])})" for j in range(start, end)]
        )
        cursor.execute(f"INSERT INTO temp_local_dt (id, dt_val) VALUES {values_str}")
        if start % 500000 == 0:
            print(f"  Insert: {start:,}/{n:,}")
            sys.stdout.flush()

    print(f"  Insert: {n:,}/{n:,}")
    print("Bulk UPDATE (single JOIN query)...")
    sys.stdout.flush()
    tu = time_mod.time()
    cursor.execute(
        "UPDATE usgs u JOIN temp_local_dt t ON u.us_id = t.id SET u.us_lt = t.dt_val"
    )
    cursor.execute("DROP TABLE temp_local_dt")
    print(f"  UPDATE done in {time_mod.time()-tu:.1f}s")

    # ==================== VERIFY ====================
    print("\nVerification from DB:")
    sys.stdout.flush()
    cursor.execute("""
        SELECT
            CASE
                WHEN us_lt <= 10 THEN '0-10 min'
                WHEN us_lt <= 60 THEN '10-60 min'
                WHEN us_lt <= 360 THEN '1-6 hours'
                WHEN us_lt <= 1440 THEN '6-24 hours'
                WHEN us_lt <= 10080 THEN '1-7 days'
                WHEN us_lt <= 43200 THEN '7-30 days'
                WHEN us_lt <= 525600 THEN '1-12 months'
                WHEN us_lt <= 5256000 THEN '1-10 years'
                WHEN us_lt < 29000000 THEN '10+ years'
                ELSE 'no local M4+'
            END as time_range,
            COUNT(*) as cnt,
            ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM usgs), 1) as pct
        FROM usgs
        GROUP BY 1
        ORDER BY MIN(us_lt)
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:15s}: {row[1]:>10,} ({row[2]}%)")
    sys.stdout.flush()

    cursor.close()
    conn.close()
    total = time_mod.time() - t0
    print(f"\nTotal time: {total:.1f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
