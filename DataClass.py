import requests
import pandas as pd
import mysql.connector
import datetime
import csv
import numpy as np
import torch
import threading
import config


class DataC():
    """Database connection-centric class: single persistent connection with thread safety."""

    # Class-level lock for thread safety
    _lock = threading.RLock()

    def __init__(self):
        self.yr_max = 55
        self.mt_max = 12
        self.x_max = 180
        self.y_max = 360
        self.m_max =  91
        self.d_max =  75
        self.t_max = 150
        self.data = []
        self.train = []
        self.valid = []
        self.n = 0.8

        # Single persistent connection
        self.mydb = None
        self.mycursor = None

        self._connect_db()
        self.fname =  "./data/latest.csv"

        # Ensure predictions table exists
        self._create_predictions_table()

    def _connect_db(self):
        """Establish database connection. Only called once or after connection dies."""
        with self._lock:
            try:
                # Clean up any existing connection first
                self._close_connection()

                self.mydb = mysql.connector.connect(
                    host=config.DB_HOST,
                    user=config.DB_USER,
                    password=config.DB_PASS,
                    database=config.DB_NAME,
                    autocommit=False,
                    connection_timeout=30,
                    ssl_disabled=True,
                    use_pure=True
                )
                self.mycursor = self.mydb.cursor()
                print("Database connection established")
            except Exception as e:
                print(f"Database connection error: {e}")
                self.mydb = None
                self.mycursor = None
                raise

    def _close_connection(self):
        """Safely close existing connection."""
        try:
            if self.mycursor:
                self.mycursor.close()
        except:
            pass
        try:
            if self.mydb:
                self.mydb.close()
        except:
            pass
        self.mydb = None
        self.mycursor = None

    def _ensure_connection(self):
        """Ensure connection is alive, reconnect if dead. Returns True if connected."""
        with self._lock:
            if self.mydb is None:
                self._connect_db()
                return self.mydb is not None

            try:
                # ping() will reconnect if connection is lost
                self.mydb.ping(reconnect=True, attempts=2, delay=1)
                # Recreate cursor after ping (old cursor may be invalid)
                if self.mycursor is None or not self.mydb.is_connected():
                    self.mycursor = self.mydb.cursor()
                return True
            except Exception as e:
                print(f"Connection ping failed: {e}, reconnecting...")
                try:
                    self._connect_db()
                    return self.mydb is not None
                except:
                    return False

    def _safe_execute(self, sql, params=None, many=False):
        """Execute SQL with automatic reconnection on failure.

        Thread-safe: uses lock to ensure single access to connection.
        Connection-centric: reuses single connection, only reconnects on failure.
        """
        with self._lock:
            max_retries = 2
            last_error = None

            for attempt in range(max_retries):
                try:
                    # Ensure we have a valid connection
                    if not self._ensure_connection():
                        raise Exception("Could not establish database connection")

                    # Execute the query
                    if many:
                        self.mycursor.executemany(sql, params)
                    elif params:
                        self.mycursor.execute(sql, params)
                    else:
                        self.mycursor.execute(sql)
                    return True

                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Connection-related errors that warrant reconnection
                    is_connection_error = any(x in error_str for x in [
                        'nonetype', 'not connected', 'unread_result', 'bytearray',
                        'weakly-referenced', 'connection not available', 'lost connection',
                        'mysql connection', 'cursor is not', 'server has gone away',
                        '2006', '2013', '2055'
                    ])

                    if attempt < max_retries - 1 and is_connection_error:
                        print(f"Query failed (attempt {attempt + 1}): {e}, reconnecting...")
                        try:
                            self._connect_db()
                        except:
                            pass
                        continue
                    else:
                        raise

            if last_error:
                raise last_error

    def _create_predictions_table(self):
        """Create predictions table if it doesn't exist"""
        sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            pr_id INT AUTO_INCREMENT PRIMARY KEY,
            pr_timestamp DATETIME NOT NULL,
            pr_lat_predicted INT,
            pr_lon_predicted INT,
            pr_dt_predicted INT,
            pr_mag_predicted INT,
            pr_lat_actual INT,
            pr_lon_actual INT,
            pr_dt_actual INT,
            pr_mag_actual INT,
            pr_actual_id INT,
            pr_actual_time DATETIME,
            pr_diff_lat INT,
            pr_diff_lon INT,
            pr_diff_dt INT,
            pr_diff_mag FLOAT,
            pr_verified BOOLEAN DEFAULT FALSE,
            pr_correct BOOLEAN DEFAULT NULL,
            pr_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_timestamp (pr_timestamp),
            INDEX idx_verified (pr_verified)
        ) ENGINE=InnoDB
        """
        try:
            self._safe_execute(sql)
            self.mydb.commit()
            # Add columns if they don't exist (for migration)
            self._migrate_predictions_table()
        except Exception as e:
            print(f"Error creating predictions table: {e}")

    def _migrate_predictions_table(self):
        """Add new columns to existing predictions table"""
        migrations = [
            "ALTER TABLE predictions ADD COLUMN pr_dt_predicted INT AFTER pr_lon_predicted",
            "ALTER TABLE predictions ADD COLUMN pr_mag_predicted INT AFTER pr_dt_predicted",
            "ALTER TABLE predictions ADD COLUMN pr_dt_actual INT AFTER pr_lon_actual",
            "ALTER TABLE predictions ADD COLUMN pr_mag_actual INT AFTER pr_dt_actual",
            "ALTER TABLE predictions ADD COLUMN pr_diff_lon INT AFTER pr_diff_lat",
            "ALTER TABLE predictions ADD COLUMN pr_diff_dt INT AFTER pr_diff_lon",
            "ALTER TABLE predictions ADD COLUMN pr_diff_mag FLOAT AFTER pr_diff_dt",
            "ALTER TABLE predictions ADD COLUMN pr_place VARCHAR(255) AFTER pr_mag_predicted",
        ]
        for sql in migrations:
            try:
                self._safe_execute(sql)
                self.mydb.commit()
            except Exception as e:
                # Column likely already exists
                pass

    def save_prediction(self, lat_predicted, lon_predicted=None, dt_predicted=None, mag_predicted=None, place=None):
        """Save a new prediction to database

        Args:
            lat_predicted: Encoded latitude (0-180)
            lon_predicted: Encoded longitude (0-360)
            dt_predicted: Time difference in minutes (0-150)
            mag_predicted: Encoded magnitude (0-91, actual = value/10)
            place: Predicted location name (from reverse geocoding)
        """
        sql = """
        INSERT INTO predictions (pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted, pr_place)
        VALUES (NOW(), %s, %s, %s, %s, %s)
        """
        try:
            self._safe_execute(sql, (lat_predicted, lon_predicted, dt_predicted, mag_predicted, place))
            self.mydb.commit()
            return self.mycursor.lastrowid
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return None

    def get_unverified_predictions(self, older_than_hours=24):
        """Get predictions that haven't been verified yet"""
        sql = """
        SELECT pr_id, pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted
        FROM predictions
        WHERE pr_verified = FALSE
        AND pr_timestamp < DATE_SUB(NOW(), INTERVAL %s HOUR)
        ORDER BY pr_timestamp ASC
        """
        try:
            self._safe_execute(sql, (older_than_hours,))
            return self.mycursor.fetchall()
        except Exception as e:
            print(f"Error getting unverified predictions: {e}")
            return []

    def get_earthquakes_in_window(self, start_time, end_time, min_mag=4.0):
        """Get actual earthquakes within a time window"""
        sql = """
        SELECT us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place
        FROM usgs
        WHERE us_datetime BETWEEN %s AND %s
        AND us_mag >= %s
        AND us_type = 'earthquake'
        ORDER BY us_datetime ASC
        """
        try:
            self._safe_execute(sql, (start_time, end_time, min_mag))
            return self.mycursor.fetchall()
        except Exception as e:
            print(f"Error getting earthquakes in window: {e}")
            return []

    def verify_prediction(self, pr_id, actual_id, actual_lat, actual_lon, actual_dt, actual_mag, actual_time, diff_lat, diff_lon, diff_dt, diff_mag, correct):
        """Update prediction with verification result

        Args:
            pr_id: Prediction ID
            actual_id: USGS earthquake ID
            actual_lat: Actual latitude (encoded 0-180)
            actual_lon: Actual longitude (encoded 0-360)
            actual_dt: Actual time difference in minutes
            actual_mag: Actual magnitude (encoded 0-91)
            actual_time: Actual earthquake datetime
            diff_lat: Difference in latitude degrees
            diff_lon: Difference in longitude degrees
            diff_dt: Difference in time prediction minutes
            diff_mag: Difference in magnitude
            correct: Whether prediction was correct
        """
        sql = """
        UPDATE predictions SET
            pr_actual_id = %s,
            pr_lat_actual = %s,
            pr_lon_actual = %s,
            pr_dt_actual = %s,
            pr_mag_actual = %s,
            pr_actual_time = %s,
            pr_diff_lat = %s,
            pr_diff_lon = %s,
            pr_diff_dt = %s,
            pr_diff_mag = %s,
            pr_verified = TRUE,
            pr_correct = %s
        WHERE pr_id = %s
        """
        try:
            self._safe_execute(sql, (actual_id, actual_lat, actual_lon, actual_dt, actual_mag, actual_time, diff_lat, diff_lon, diff_dt, diff_mag, correct, pr_id))
            self.mydb.commit()
            return True
        except Exception as e:
            print(f"Error verifying prediction: {e}")
            return False

    def get_predictions_with_actuals(self, limit=20, offset=0, min_mag=4.0, filter_type=None):
        """Get predictions with matched actual earthquakes for display (paginated)

        Args:
            limit: Maximum number of predictions to return per page
            offset: Number of predictions to skip (for pagination)
            min_mag: Minimum predicted magnitude to show (default 4.0)
            filter_type: Optional filter - 'matched', 'missed', 'pending', or None for all

        Returns:
            dict with 'predictions' list and 'total' count
        """
        encoded_mag = min_mag * 10

        # Build WHERE clause based on filter
        filter_clause = "p.pr_mag_predicted >= %s"
        params = [encoded_mag]

        if filter_type == 'matched':
            filter_clause += " AND p.pr_correct = TRUE"
        elif filter_type == 'missed':
            filter_clause += " AND p.pr_verified = TRUE AND p.pr_correct = FALSE"
        elif filter_type == 'pending':
            filter_clause += " AND p.pr_verified = FALSE"

        # Get total count
        count_sql = f"SELECT COUNT(*) FROM predictions p WHERE {filter_clause}"
        try:
            self._safe_execute(count_sql, tuple(params))
            total = self.mycursor.fetchone()[0]
        except Exception as e:
            print(f"Error counting predictions: {e}")
            total = 0

        # Get paginated results (latest first)
        sql = f"""
        SELECT
            p.pr_id,
            p.pr_timestamp,
            p.pr_lat_predicted - 90 as predicted_lat,
            p.pr_lon_predicted - 180 as predicted_lon,
            p.pr_dt_predicted as predicted_dt,
            p.pr_mag_predicted / 10.0 as predicted_mag,
            p.pr_place as predicted_place,
            p.pr_lat_actual - 90 as actual_lat,
            p.pr_lon_actual - 180 as actual_lon,
            p.pr_dt_actual as actual_dt,
            p.pr_mag_actual / 10.0 as actual_mag,
            p.pr_diff_lat as diff_lat,
            p.pr_diff_lon as diff_lon,
            p.pr_diff_dt as diff_dt,
            p.pr_diff_mag as diff_mag,
            p.pr_verified,
            p.pr_correct,
            p.pr_actual_time,
            u.us_place as actual_place
        FROM predictions p
        LEFT JOIN usgs u ON p.pr_actual_id = u.us_id
        WHERE {filter_clause}
        ORDER BY p.pr_timestamp DESC
        LIMIT %s OFFSET %s
        """
        try:
            self._safe_execute(sql, tuple(params + [limit, offset]))
            columns = ['id', 'prediction_time', 'predicted_lat', 'predicted_lon', 'predicted_dt', 'predicted_mag', 'predicted_place',
                      'actual_lat', 'actual_lon', 'actual_dt', 'actual_mag', 'diff_lat', 'diff_lon', 'diff_dt', 'diff_mag',
                      'verified', 'correct', 'actual_time', 'actual_place']
            rows = self.mycursor.fetchall()
            predictions = [dict(zip(columns, row)) for row in rows]
            return {'predictions': predictions, 'total': total}
        except Exception as e:
            print(f"Error getting predictions with actuals: {e}")
            return {'predictions': [], 'total': 0}

    def get_latest_prediction(self, min_mag=4.0):
        """Get the most recent prediction with magnitude >= min_mag"""
        sql = """
        SELECT
            pr_id,
            pr_timestamp,
            pr_lat_predicted - 90 as predicted_lat,
            pr_lon_predicted - 180 as predicted_lon,
            pr_dt_predicted as predicted_dt,
            pr_mag_predicted / 10.0 as predicted_mag,
            pr_place,
            pr_lat_actual - 90 as actual_lat,
            pr_lon_actual - 180 as actual_lon,
            pr_dt_actual as actual_dt,
            pr_mag_actual / 10.0 as actual_mag,
            pr_diff_lat,
            pr_diff_lon,
            pr_diff_dt,
            pr_diff_mag,
            pr_verified,
            pr_correct
        FROM predictions
        WHERE pr_mag_predicted >= %s
        ORDER BY pr_timestamp DESC
        LIMIT 1
        """
        try:
            # min_mag stored as encoded (mag * 10), so multiply threshold
            self._safe_execute(sql, (min_mag * 10,))
            row = self.mycursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'timestamp': row[1].isoformat() if row[1] else None,
                    'predicted_lat': row[2],
                    'predicted_lon': row[3],
                    'predicted_dt': row[4],
                    'predicted_mag': float(row[5]) if row[5] else None,
                    'predicted_place': row[6],
                    'actual_lat': row[7],
                    'actual_lon': row[8],
                    'actual_dt': row[9],
                    'actual_mag': float(row[10]) if row[10] else None,
                    'diff_lat': row[11],
                    'diff_lon': row[12],
                    'diff_dt': row[13],
                    'diff_mag': float(row[14]) if row[14] else None,
                    'verified': bool(row[15]),
                    'correct': bool(row[16]) if row[16] is not None else None
                }
            return None
        except Exception as e:
            print(f"Error getting latest prediction: {e}")
            return None

    def get_recent_earthquakes(self, limit=20, min_mag=4.0):
        """Get recent actual earthquakes from database"""
        sql = """
        SELECT
            us_id,
            us_datetime,
            us_x - 90 as lat,
            us_y - 180 as lon,
            us_mag,
            us_dep,
            us_place
        FROM usgs
        WHERE us_mag >= %s AND us_type = 'earthquake'
        ORDER BY us_datetime DESC
        LIMIT %s
        """
        try:
            self._safe_execute(sql, (min_mag, limit))
            columns = ['id', 'time', 'lat', 'lon', 'mag', 'depth', 'place']
            rows = self.mycursor.fetchall()
            result = []
            for row in rows:
                d = dict(zip(columns, row))
                if d['time']:
                    d['time'] = d['time'].isoformat()
                result.append(d)
            return result
        except Exception as e:
            print(f"Error getting recent earthquakes: {e}")
            return []

    def update_prediction_match(self, pr_id, earthquake_id, earthquake_lat, earthquake_lon, earthquake_mag, earthquake_time, distance):
        """Update prediction with real-time match data when a match is detected on the frontend

        Args:
            pr_id: Prediction ID to update
            earthquake_id: USGS earthquake ID that matched
            earthquake_lat: Actual earthquake latitude (already decoded)
            earthquake_lon: Actual earthquake longitude (already decoded)
            earthquake_mag: Actual earthquake magnitude
            earthquake_time: Actual earthquake time (ISO string)
            distance: Circular distance in degrees
        """
        try:
            # First get the prediction data to calculate differences
            sql = """SELECT pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted
                     FROM predictions WHERE pr_id = %s"""
            self._safe_execute(sql, (pr_id,))
            row = self.mycursor.fetchone()

            if not row:
                print(f"Prediction {pr_id} not found")
                return False

            pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted = row

            # Convert earthquake coords to encoded format (for storage consistency)
            eq_lat_encoded = int(earthquake_lat + 90)
            eq_lon_encoded = int(earthquake_lon + 180)
            eq_mag_encoded = int(earthquake_mag * 10)

            # Calculate differences
            diff_lat = abs(pr_lat_predicted - eq_lat_encoded) if pr_lat_predicted else None
            diff_lon = abs(pr_lon_predicted - eq_lon_encoded) if pr_lon_predicted else None
            if diff_lon:
                diff_lon = min(diff_lon, 360 - diff_lon)

            # Parse earthquake time and calculate dt
            from datetime import datetime
            if isinstance(earthquake_time, str):
                eq_time = datetime.fromisoformat(earthquake_time.replace('Z', '+00:00'))
            else:
                eq_time = earthquake_time

            actual_dt = int((eq_time.replace(tzinfo=None) - pr_timestamp).total_seconds() / 60)
            diff_dt = abs(pr_dt_predicted - actual_dt) if pr_dt_predicted else None

            diff_mag = abs((pr_mag_predicted / 10.0) - earthquake_mag) if pr_mag_predicted else None

            # A match within 15 degrees is considered correct
            correct = distance <= 15

            sql = """
            UPDATE predictions SET
                pr_actual_id = %s,
                pr_lat_actual = %s,
                pr_lon_actual = %s,
                pr_dt_actual = %s,
                pr_mag_actual = %s,
                pr_actual_time = %s,
                pr_diff_lat = %s,
                pr_diff_lon = %s,
                pr_diff_dt = %s,
                pr_diff_mag = %s,
                pr_verified = TRUE,
                pr_correct = %s
            WHERE pr_id = %s
            """
            self._safe_execute(sql, (
                earthquake_id, eq_lat_encoded, eq_lon_encoded, actual_dt, eq_mag_encoded, eq_time,
                diff_lat, diff_lon, diff_dt, diff_mag, correct, pr_id
            ))
            self.mydb.commit()
            print(f"Updated prediction {pr_id} with match: distance={distance:.1f}, correct={correct}")
            return True

        except Exception as e:
            print(f"Error updating prediction match: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_prediction_stats(self, min_mag=4.0):
        """Get prediction success statistics for predictions with magnitude >= min_mag"""
        sql = """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pr_verified = TRUE THEN 1 ELSE 0 END) as verified,
            SUM(CASE WHEN pr_correct = TRUE THEN 1 ELSE 0 END) as correct
        FROM predictions
        WHERE pr_mag_predicted >= %s
        """
        try:
            # min_mag stored as encoded (mag * 10), so multiply threshold
            self._safe_execute(sql, (min_mag * 10,))
            row = self.mycursor.fetchone()
            total = row[0] or 0
            verified = row[1] or 0
            correct = row[2] or 0
            return {
                'total_predictions': total,
                'verified_predictions': verified,
                'correct_predictions': correct,
                'success_rate': (correct / verified * 100) if verified > 0 else 0.0
            }
        except Exception as e:
            print(f"Error getting prediction stats: {e}")
            return {'total_predictions': 0, 'verified_predictions': 0, 'correct_predictions': 0, 'success_rate': 0.0}
        
    def db2File(self, min_mag):
        # Create completely fresh connection for export
        try:
            db = mysql.connector.connect(
                host=config.DB_HOST,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                ssl_disabled=True
            )
            cursor = db.cursor()

            sql = f"CALL get_data({min_mag})"
            cursor.execute(sql)
            rows = cursor.fetchall()

            cols = [i[0] for i in cursor.description] if cursor.description else []

            # Consume any remaining results
            try:
                while cursor.nextset():
                    pass
            except:
                pass

            if rows and cols:
                fp = open(self.fname, 'w')
                myFile = csv.writer(fp, lineterminator='\n')
                myFile.writerow(cols)
                myFile.writerows(rows)
                fp.close()
                print(f"Exported {len(rows)} records to CSV")

            cursor.close()
            db.close()

        except Exception as e:
            print(f"Error in db2File: {e}")
        
    def usgs2DB(self, days=1):
        """Fetch USGS data. days=1 for quick updates, days=3 for full refresh"""
        self._safe_execute("delete from usgs_tmp")

        min_magnitude = 2
        all_values = []

        # Collect data from USGS API
        for i in range(days, 0, -1):
            start_time = f"now-{i}days"
            end_time = f"now-{(i-1)}days"
            path = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_time}&endtime={end_time}&minmagnitude={min_magnitude}"

            try:
                response = requests.get(path, timeout=15)
                dataset = response.json()
                features = dataset['features']
                print(f"Fetched {len(features)} earthquakes for day -{i}")

                for feature in features:
                    code = feature['id']
                    mag = feature['properties']['mag'] or 0
                    plc = feature['properties']['place']
                    if plc:
                        plc = plc.replace("'", "").replace('"', "")
                    else:
                        plc = ""

                    dtm = pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%y/%m/%d %H:%M:%S')
                    dat = pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%y/%m/%d')
                    tim = pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%H:%M:%S')

                    typ = feature['properties']['type'] or ''
                    magType = feature['properties']['magType'] or ''

                    lon = feature['geometry']['coordinates'][0]
                    lat = feature['geometry']['coordinates'][1]
                    dep = feature['geometry']['coordinates'][2] or 0
                    if dep < 0:
                        dep = 0
                    x = lat + 90
                    y = lon + 180
                    d = dep / 10
                    m = mag * 10

                    all_values.append((code, dat, tim, dtm, lat, lon, dep, mag, plc, typ, magType, x, y, d, m))

            except Exception as e:
                print(f"Error fetching USGS data for day -{i}: {e}")

        # Batch insert all records at once
        if all_values:
            sql = """INSERT INTO usgs_tmp(us_code, us_date, us_time, us_datetime, us_lat, us_lon,
                     us_dep, us_mag, us_place, us_type, us_magType, us_x, us_y, us_d, us_m)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            try:
                self._safe_execute(sql, all_values, many=True)
                self.mydb.commit()
                print(f"Batch inserted {len(all_values)} earthquakes")
            except Exception as e:
                print(f"Batch insert error: {e}")

        # Run stored procedure to merge into main table
        try:
            self._safe_execute("call ins_quakes()")
            print("Executed ins_quakes()")
        except Exception as e:
            print(f"Error ins_quakes: {e}")

        # Calculate time differences for new records
        self._safe_execute("SELECT us_id FROM usgs WHERE us_t is null")
        rows = self.mycursor.fetchall()
        if rows:
            print(f"Calculating dt for {len(rows)} new records...")
            for row in rows:
                try:
                    self._safe_execute(f"call calc_dt({row[0]})")
                except Exception as e:
                    print(f"Error calc_dt for {row[0]}: {e}")
            self.mydb.commit()
            print("Done calculating dt")
        
    def getData(self, from_db=True, min_mag=3.9):
        """Load training data from database or CSV.

        Args:
            from_db: If True, load directly from database (faster, no CSV export needed)
                     If False, load from CSV file
            min_mag: Minimum magnitude filter (default 3.9)
        """
        if from_db:
            return self.getDataFromDB(min_mag)

        # Legacy: load from CSV
        df1 = pd.read_csv(self.fname)
        df1.fillna(0, inplace=True)
        self.data = np.array(df1.values)
        self._update_data_splits()
        return self.data

    def getDataFromDB(self, min_mag=3.9):
        """Load training data directly from database using stored procedure.

        This eliminates the need for CSV export, making data loading much faster.
        Uses get_data_fast which reads pre-computed us_t values directly.
        Columns returned: [year, month, x, y, m, d, t]
        """
        try:
            # Use fresh connection for data loading
            db = mysql.connector.connect(
                host=config.DB_HOST,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                ssl_disabled=True
            )
            cursor = db.cursor()

            # Call optimized stored procedure (uses pre-computed us_t)
            cursor.execute(f"CALL get_data_fast({min_mag})")
            rows = cursor.fetchall()

            # Consume any remaining results
            try:
                while cursor.nextset():
                    pass
            except:
                pass

            cursor.close()
            db.close()

            if rows:
                self.data = np.array(rows, dtype=np.float64)
                self.data = np.nan_to_num(self.data, 0)
                print(f"Loaded {len(self.data):,} records from database")
                self._update_data_splits()
            else:
                print("No data returned from database")
                self.data = np.array([])

            return self.data

        except Exception as e:
            print(f"Error loading from DB: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])

    def _update_data_splits(self):
        """Update train/valid splits and max values from loaded data."""
        n = int(self.n * len(self.data))
        self.train = self.data[:n]
        self.valid = self.data[n:]

        self.yr_max = int(self.data[:, 0].max()) + 1  # year 0 to 54
        self.mt_max = int(self.data[:, 1].max()) + 1
        self.x_max  = int(self.data[:, 2].max()) + 1
        self.y_max  = int(self.data[:, 3].max()) + 1
        self.m_max  = int(self.data[:, 4].max()) + 1
        self.d_max  = int(self.data[:, 5].max()) + 1
        self.t_max  = int(self.data[:, 6].max()) + 1

    def getDataHybrid(self, input_mag=2.0, target_mag=4.0):
        """Load hybrid training data: ALL earthquakes >= input_mag as context,
        but only earthquakes >= target_mag are valid training targets.

        Args:
            input_mag: Minimum magnitude for input sequences (default 2.0 for all context)
            target_mag: Minimum magnitude for training targets (default 4.0 for M4+)

        Returns:
            data array with columns: [year, month, x, y, m, d, t, is_target]
            is_target=1 for M4+ earthquakes, 0 for smaller ones
        """
        try:
            db = mysql.connector.connect(
                host=config.DB_HOST,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                ssl_disabled=True
            )
            cursor = db.cursor()

            cursor.execute(f"CALL get_data_hybrid({input_mag}, {target_mag})")
            rows = cursor.fetchall()

            try:
                while cursor.nextset():
                    pass
            except:
                pass

            cursor.close()
            db.close()

            if rows:
                self.data = np.array(rows, dtype=np.float64)
                self.data = np.nan_to_num(self.data, 0)

                # Store target indices for efficient sampling
                self.target_indices = np.where(self.data[:, 7] == 1)[0]
                target_count = len(self.target_indices)

                print(f"Loaded {len(self.data):,} records (M{input_mag}+)")
                print(f"  Training targets (M{target_mag}+): {target_count:,} ({100*target_count/len(self.data):.1f}%)")

                self._update_data_splits()
            else:
                print("No data returned from database")
                self.data = np.array([])
                self.target_indices = np.array([])

            return self.data

        except Exception as e:
            print(f"Error loading hybrid data: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
        
    def getSizes(self):
        if (len(self.data)<1):
            print("Array size, please call getData first")
            return "Error " 
        # print(f" Yr min:  {self.data[:,0].min()} max : {self.data[:,0].max()}")
        # print(f" Mt min:  {self.data[:,1].min()} max : {self.data[:,1].max()}")
        # print(f" x  min:  {self.data[:,2].min()} max : {self.data[:,2].max()}")
        # print(f" y  min:  {self.data[:,3].min()} max : {self.data[:,3].max()}")
        # print(f" m  min:  {self.data[:,4].min()} max : {self.data[:,4].max()}")
        # print(f" d  min:  {self.data[:,5].min()} max : {self.data[:,5].max()}")
        # print(f" t  min:  {self.data[:,6].min()} max : {self.data[:,6].max()}")
        sizes = {
            'yr_size' : self.yr_max, 
            'mt_size' : self.mt_max, 
            'x_size' : self.x_max, 
            'y_size' : self.y_max, 
            'm_size' : self.m_max, 
            'd_size' : self.d_max, 
            't_size' : self.t_max
        }
        return sizes    

    def getScaledData(self):
        data = self.getData()
        # I checked data and values up until this point
        #
        # year [0 54] Some years are active then others
        # month [1 12] some months are active than other, July is the most active month for sure
        # lat [6 177] most activity around 130 degree (range is [0 180])
        # lon [0 360] most activity around  70 degree (range is [0 360])
        # mag [0 9] 2,3,4 dominate the data
        # dep [0 70] 1 and 10 dominates
        # dt  [0 150] in minutes, nice curve
        self.yr_max = data[:, 0].max()  # year 0 to 54
        self.mt_max = data[:, 1].max()  
        self.x_max  = data[:, 2].max()  
        self.y_max  = data[:, 3].max()  
        self.m_max  = data[:, 4].max()  
        self.d_max  = data[:, 5].max()  
        self.t_max  = data[:, 6].max()
        return data
    
    def getBatch(self, B, T, split, col):

        if (len(self.data)<1):
            self.getData()

        data_ = self.train if split=='train' else self.valid

        dataT = torch.from_numpy(data_[:, col]) # latitude
        data_ = torch.from_numpy(data_)

        ix = torch.randint(len(data_)-T, (B,))
        # print(f"ix {ix}")
        # ix = range(B)

        x = torch.stack([data_[i:i+T    ] for i in ix])
        y = torch.stack([dataT[i+1:i+T+1] for i in ix])

        # print(f"x shape {x.shape} y shape {y.shape}")
        # print(f"x data  {x}")
        # print(f"y data  {y}")
        # print(f"{1/0}")
        # x => B, T, F
        # y => B, T

        return x.long(), y.long()

    def getBatchMulti(self, B, T, split):
        """Get batch with all 4 targets for multi-target training.

        Returns:
            x: Input sequences [B, T, 7]
            targets: Dict with 'lat', 'lon', 'dt', 'mag' tensors [B]
                - lat: column 2 (latitude 0-180)
                - lon: column 3 (longitude 0-360)
                - mag: column 4 (magnitude 0-91)
                - dt:  column 6 (time difference 0-150, clamped)
        """
        if len(self.data) < 1:
            self.getData()

        data_ = self.train if split == 'train' else self.valid
        data_tensor = torch.from_numpy(data_)

        ix = torch.randint(len(data_) - T - 1, (B,))

        # x is the input sequence
        x = torch.stack([data_tensor[i:i+T] for i in ix])

        # Get next position (T+1) for all samples - this is what we predict
        next_pos = torch.stack([data_tensor[i+T] for i in ix])

        # Extract each target column from the next position
        # Clamp values to valid model output ranges
        targets = {
            'lat': torch.clamp(next_pos[:, 2], 0, 180).long(),  # latitude (0-180)
            'lon': torch.clamp(next_pos[:, 3], 0, 360).long(),  # longitude (0-360)
            'mag': torch.clamp(next_pos[:, 4], 0, 91).long(),   # magnitude (0-91)
            'dt':  torch.clamp(next_pos[:, 6], 0, 150).long(),  # time difference (0-150)
        }

        return x.long(), targets

    def getBatchHybrid(self, B, T, split):
        """Get batch for hybrid training: input has ALL earthquakes, targets are M4+ only.

        This method requires getDataHybrid() to be called first.
        Samples sequences where the target position is a M4+ earthquake.

        Returns:
            x: Input sequences [B, T, 7] - includes all earthquakes as context
            targets: Dict with 'lat', 'lon', 'dt', 'mag' tensors [B] - only M4+ targets
        """
        if len(self.data) < 1 or not hasattr(self, 'target_indices'):
            raise ValueError("Call getDataHybrid() first to load hybrid data")

        # Get train/valid split
        n_train = int(self.n * len(self.data))

        if split == 'train':
            # Target indices that are in training set and have enough context
            valid_targets = self.target_indices[
                (self.target_indices >= T) & (self.target_indices < n_train)
            ]
        else:
            # Target indices that are in validation set
            valid_targets = self.target_indices[
                (self.target_indices >= n_train + T)
            ]

        if len(valid_targets) < B:
            raise ValueError(f"Not enough M4+ targets in {split} set (need {B}, have {len(valid_targets)})")

        # Sample B random M4+ target positions
        sample_idx = torch.randint(len(valid_targets), (B,))
        target_positions = valid_targets[sample_idx.numpy()]

        # Ensure target_positions is always iterable
        if not hasattr(target_positions, '__len__') or len(target_positions.shape) == 0:
            target_positions = [target_positions.item()]

        # Get input sequences (T positions before each target)
        # Only use first 7 columns (exclude is_target column)
        data_tensor = torch.from_numpy(self.data[:, :7])

        x = torch.stack([data_tensor[int(pos)-T:int(pos)] for pos in target_positions])

        # Get target values from the M4+ positions
        next_pos = torch.stack([data_tensor[int(pos)] for pos in target_positions])

        targets = {
            'lat': torch.clamp(next_pos[:, 2], 0, 180).long(),
            'lon': torch.clamp(next_pos[:, 3], 0, 360).long(),
            'mag': torch.clamp(next_pos[:, 4], 0, 91).long(),
            'dt':  torch.clamp(next_pos[:, 6], 0, 150).long(),
        }

        return x.long(), targets

    def getLast(self, B, T, split, col):
        """Get the LAST sequence of T earthquakes for inference prediction.

        Args:
            B: Batch size (typically 1 for inference)
            T: Sequence length
            split: 'train' or 'val'
            col: Column index for target (not used in hybrid mode)

        Returns:
            x: Last T earthquakes [B, T, 7]
            y: Target values [B]
        """
        if len(self.data) < 1:
            self.getData()

        # Use all data for getting the latest context
        # Only use first 7 columns (exclude is_target flag if present)
        num_cols = min(7, self.data.shape[1])
        data_ = torch.from_numpy(self.data[:, :num_cols])

        # Get the LAST T records for prediction
        start_idx = len(data_) - T
        if start_idx < 0:
            raise ValueError(f"Not enough data: have {len(data_)}, need {T}")

        x = data_[start_idx:].unsqueeze(0)  # [1, T, 7]

        # For y, return the last record's target column (used for reference only)
        y = data_[-1, col].unsqueeze(0)  # [1]

        return x.long(), y.long()
    

    def closeAll(self):
        """Close the database connection."""
        self._close_connection()
        