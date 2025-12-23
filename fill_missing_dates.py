#!/usr/bin/env python3
"""
Script to find missing dates in USGS table and fill them from USGS API
"""

import requests
from datetime import datetime, timedelta
import mysql.connector
import config
import time

def get_db_connection():
    return mysql.connector.connect(
        host=config.DB_HOST,
        user=config.DB_USER,
        password=config.DB_PASS,
        database=config.DB_NAME,
        autocommit=True
    )

def get_existing_dates(cursor, start_date, end_date):
    """Get all dates that have data in the database"""
    sql = """
    SELECT DISTINCT DATE(us_datetime) as dt
    FROM usgs
    WHERE us_datetime >= %s AND us_datetime < %s
    ORDER BY dt
    """
    cursor.execute(sql, (start_date, end_date))
    return set(row[0] for row in cursor.fetchall())

def find_missing_dates(start_date, end_date, existing_dates):
    """Find dates that are missing from the database"""
    missing = []
    current = start_date
    while current < end_date:
        if current not in existing_dates:
            missing.append(current)
        current += timedelta(days=1)
    return missing

def fetch_usgs_data(start_date, end_date, min_magnitude=2.0):
    """Fetch earthquake data from USGS API for a date range"""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        'format': 'geojson',
        'starttime': start_date.strftime('%Y-%m-%d'),
        'endtime': end_date.strftime('%Y-%m-%d'),
        'minmagnitude': min_magnitude,
        'orderby': 'time'
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Error fetching USGS data: {e}")
        return None

def insert_earthquake(cursor, eq):
    """Insert a single earthquake into the database"""
    props = eq['properties']
    coords = eq['geometry']['coordinates']

    # Parse time
    eq_time = datetime.fromtimestamp(props['time'] / 1000)

    # Calculate encoded values
    lat = coords[1]
    lon = coords[0]
    depth = coords[2] if coords[2] else 0
    mag = props['mag'] if props['mag'] else 0

    # Encode values (matching existing schema)
    x = int(lat + 90)  # 0-180
    y = int(lon + 180)  # 0-360
    d = min(int(depth), 74)  # 0-74
    m = int(mag * 10)  # 0-91

    sql = """
    INSERT IGNORE INTO usgs
    (us_code, us_date, us_time, us_datetime, us_lat, us_lon, us_dep, us_mag,
     us_magtype, us_type, us_x, us_y, us_d, us_m, us_place)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        cursor.execute(sql, (
            props.get('code', ''),
            eq_time.date(),
            eq_time.time(),
            eq_time,
            lat,
            lon,
            depth,
            mag,
            props.get('magType', ''),
            props.get('type', 'earthquake'),
            x, y, d, m,
            props.get('place', '')[:100] if props.get('place') else ''
        ))
        return True
    except Exception as e:
        print(f"  Error inserting: {e}")
        return False

def main():
    print("=" * 60)
    print("USGS Missing Dates Checker and Filler")
    print("=" * 60)

    db = get_db_connection()
    cursor = db.cursor()

    # Check for missing dates in the last 2 years (more recent data is more important)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=730)  # 2 years back

    print(f"\nChecking for missing dates from {start_date} to {end_date}...")

    # Get existing dates
    existing_dates = get_existing_dates(cursor, start_date, end_date)
    print(f"Found {len(existing_dates)} dates with data")

    # Find missing dates
    missing_dates = find_missing_dates(start_date, end_date, existing_dates)

    if not missing_dates:
        print("\nNo missing dates found! Database is complete.")
        return

    print(f"\nFound {len(missing_dates)} missing dates:")
    for d in missing_dates[:20]:  # Show first 20
        print(f"  - {d}")
    if len(missing_dates) > 20:
        print(f"  ... and {len(missing_dates) - 20} more")

    # Fill missing dates
    print(f"\nFilling missing dates...")
    total_inserted = 0

    # Process in batches of 7 days to avoid API limits
    i = 0
    while i < len(missing_dates):
        batch_start = missing_dates[i]

        # Find consecutive missing dates (max 7 days)
        batch_end = batch_start
        while i + 1 < len(missing_dates) and missing_dates[i + 1] - batch_end <= timedelta(days=1):
            i += 1
            batch_end = missing_dates[i]
            if (batch_end - batch_start).days >= 6:
                break

        # Fetch data for this range
        fetch_end = batch_end + timedelta(days=1)
        print(f"\n  Fetching {batch_start} to {batch_end}...", end=" ")

        data = fetch_usgs_data(batch_start, fetch_end)

        if data and 'features' in data:
            count = 0
            for eq in data['features']:
                if insert_earthquake(cursor, eq):
                    count += 1

            db.commit()
            print(f"inserted {count} earthquakes")
            total_inserted += count
        else:
            print("no data or error")

        i += 1

        # Rate limiting - be nice to USGS API
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"Complete! Total earthquakes inserted: {total_inserted}")
    print(f"{'=' * 60}")

    cursor.close()
    db.close()

if __name__ == '__main__':
    main()
