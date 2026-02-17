import requests
import pandas as pd
import mysql.connector
import datetime
import csv
import uuid
import numpy as np
import torch
import threading
import config


class DataC():
    """Database connection-centric class: single persistent connection with thread safety."""

    # Class-level lock for thread safety
    _lock = threading.RLock()

    # Log-scale bin midpoints for converting bin index → minutes
    # Bin 0: 0min, Bin 1: 1min, Bin 2: 2-3min, Bin 3: 4-7min, Bin 4: 8-15min,
    # Bin 5: 16-31min, Bin 6: 32-63min, Bin 7: 64-127min, Bin 8: 128-255min, Bin 9: 256+min
    LOG_BIN_MIDPOINTS = [0, 1, 2, 5, 11, 22, 45, 90, 181, 308]

    def __init__(self):
        # Embedding size constants (max index value, NOT count)
        # Embedding layer needs size = max_index + 1
        self.yr_max = 60       # Years 0-60 (1970-2030) → 61 embeddings
        self.mt_max = 11       # Months 0-11 (converted from 1-12) → 12 embeddings
        self.x_max = 180       # Latitude 0-180 (encoded: lat+90) → 181 embeddings
        self.y_max = 360       # Longitude 0-360 (encoded: lon+180) → 361 embeddings
        self.m_max = 91        # Magnitude 0-91 (encoded: mag*10) → 92 embeddings
        self.d_max = 200       # Depth 0-200 km → 201 embeddings (captures subduction zones)
        self.t_max = 9         # Log-scale bins 0-9 → 10 embeddings
        self.t_raw_max = 360   # Raw minutes cap before log binning
        self.lt_max = 25       # Local dt log-scale bins 0-25 → 26 embeddings
        self.lt_raw_max = 29000000  # ~55 years in minutes (no upper cap)
        self.hr_max = 23       # Hour of day 0-23 → 24 embeddings
        self.doy_max = 365     # Day of year 0-365 → 366 embeddings
        self.mp_max = 29       # Moon phase 0-29 → 30 embeddings (0=new, 15=full)
        self.md_max = 9        # Moon distance bin 0-9 → 10 embeddings (0=perigee, 9=apogee)
        self.data = []
        self.train = []
        self.valid = []
        self.n = 0.8

        # Single persistent connection
        self.mydb = None
        self.mycursor = None

        self._connect_db()
        self.fname =  "./data/latest.csv"

        # Ensure tables exist
        self._create_predictions_table()
        self._create_email_alerts_table()

    @staticmethod
    def _log_bin_tensor(t, n_bins=10):
        """Convert raw minutes tensor to log-scale bin indices.

        Matches Omori's law (aftershock rate ~ 1/t) — gives equal
        information density per bin instead of wasting 95% of embeddings
        on the flat tail.

        Bin 0: 0 min, Bin 1: 1 min, Bin 2: 2-3 min, Bin 3: 4-7 min,
        Bin 4: 8-15 min, Bin 5: 16-31 min, Bin 6: 32-63 min,
        Bin 7: 64-127 min, Bin 8: 128-255 min, Bin 9: 256+ min
        """
        result = torch.zeros(t.shape, dtype=t.dtype, device=t.device)
        mask = t > 0
        if mask.any():
            log_vals = torch.log2(t[mask].float().clamp(min=1))
            result[mask] = torch.clamp(log_vals.floor() + 1, max=n_bins - 1).to(t.dtype)
        return result

    @staticmethod
    def _compute_moon_features(unix_timestamps):
        """Compute moon phase and distance bins from unix timestamps.

        Uses simple orbital mechanics — no external dependencies.
        Vectorized with numpy for fast computation on 1M+ records.

        Returns:
            moon_phase: int array 0-29 (0=new moon, 15=full moon)
            moon_dist_bin: int array 0-9 (0=perigee/closest, 9=apogee/farthest)
        """
        SYNODIC_PERIOD = 29.530588853       # days (new moon to new moon)
        REF_NEW_MOON = 947182440            # Jan 6, 2000 18:14 UTC (known new moon)
        ANOMALISTIC_PERIOD = 27.554551      # days (perigee to perigee)
        REF_PERIGEE = 948019200             # Jan 16, 2000 ~00:00 UTC (approximate perigee)

        ts = np.asarray(unix_timestamps, dtype=np.float64)

        # Moon phase: 0-29 based on synodic cycle
        days_since_new = (ts - REF_NEW_MOON) / 86400.0
        moon_phase = ((days_since_new % SYNODIC_PERIOD) / SYNODIC_PERIOD * 30).astype(np.int32) % 30

        # Moon distance: approximate using eccentricity model
        # Distance varies from ~356,500 km (perigee) to ~406,700 km (apogee)
        days_since_perigee = (ts - REF_PERIGEE) / 86400.0
        anomaly = 2 * np.pi * (days_since_perigee % ANOMALISTIC_PERIOD) / ANOMALISTIC_PERIOD
        distance_km = 384400 * (1 - 0.0549 * np.cos(anomaly))
        moon_dist_bin = np.clip(((distance_km - 356500) / (406700 - 356500) * 10).astype(np.int32), 0, 9)

        return moon_phase, moon_dist_bin

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
                # ALWAYS recreate cursor after ping - ping(reconnect=True) may have
                # internally replaced the connection, making old cursor invalid
                # (causes "Bad file descriptor" errors if we reuse stale cursor)
                try:
                    if self.mycursor:
                        self.mycursor.close()
                except:
                    pass
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

    def _safe_fetch(self, sql, params=None, fetch_one=False):
        """Execute SQL and fetch results in one atomic operation.

        Thread-safe: keeps lock held during execute AND fetch to prevent
        cursor invalidation between the two operations.

        Args:
            sql: SQL query string
            params: Query parameters (tuple)
            fetch_one: If True, return single row; if False, return all rows

        Returns:
            Single row (if fetch_one) or list of rows (if not fetch_one)
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
                    if params:
                        self.mycursor.execute(sql, params)
                    else:
                        self.mycursor.execute(sql)

                    # Fetch results immediately while still holding lock
                    if fetch_one:
                        return self.mycursor.fetchone()
                    else:
                        return self.mycursor.fetchall()

                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Connection-related errors that warrant reconnection
                    is_connection_error = any(x in error_str for x in [
                        'nonetype', 'not connected', 'unread_result', 'bytearray',
                        'weakly-referenced', 'connection not available', 'lost connection',
                        'mysql connection', 'cursor is not', 'server has gone away',
                        '2006', '2013', '2055', 'bad file descriptor'
                    ])

                    if attempt < max_retries - 1 and is_connection_error:
                        print(f"Fetch failed (attempt {attempt + 1}): {e}, reconnecting...")
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

    def _create_email_alerts_table(self):
        """Create email_alerts table if it doesn't exist"""
        sql = """
        CREATE TABLE IF NOT EXISTS email_alerts (
            ea_id INT AUTO_INCREMENT PRIMARY KEY,
            ea_email VARCHAR(255) NOT NULL,
            ea_lat DECIMAL(7,4) NOT NULL,
            ea_lon DECIMAL(7,4) NOT NULL,
            ea_radius_km INT NOT NULL DEFAULT 500,
            ea_active BOOLEAN NOT NULL DEFAULT FALSE,
            ea_verified BOOLEAN NOT NULL DEFAULT FALSE,
            ea_verify_token VARCHAR(36) NOT NULL,
            ea_unsubscribe_token VARCHAR(36) NOT NULL,
            ea_last_notified DATETIME DEFAULT NULL,
            ea_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY idx_email_location (ea_email, ea_lat, ea_lon),
            INDEX idx_active_verified (ea_active, ea_verified),
            INDEX idx_unsub_token (ea_unsubscribe_token),
            INDEX idx_verify_token (ea_verify_token)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        try:
            self._safe_execute(sql)
            self.mydb.commit()
        except Exception as e:
            print(f"Error creating email_alerts table: {e}")

        # Migrations for existing tables
        migrations = [
            "ALTER TABLE email_alerts ADD COLUMN ea_verified BOOLEAN NOT NULL DEFAULT FALSE AFTER ea_active",
            "ALTER TABLE email_alerts ADD COLUMN ea_verify_token VARCHAR(36) NOT NULL DEFAULT '' AFTER ea_verified",
            "ALTER TABLE email_alerts ADD INDEX idx_verify_token (ea_verify_token)",
            "UPDATE email_alerts SET ea_verified = TRUE WHERE ea_active = TRUE AND ea_verified = FALSE",
        ]
        for m in migrations:
            try:
                self._safe_execute(m)
                self.mydb.commit()
            except Exception:
                pass

    def add_email_alert(self, email, lat, lon, radius_km=500):
        """Subscribe an email to earthquake alerts for a location.

        Creates an UNVERIFIED, INACTIVE subscription. User must click
        the verification link in the email to activate it.

        Returns:
            dict with ea_id, verify_token, unsub_token on success
            None if already active+verified for this location
            'pending' if already pending verification
        """
        verify_token = str(uuid.uuid4())
        unsub_token = str(uuid.uuid4())
        try:
            with self._lock:
                existing = self._safe_fetch(
                    "SELECT ea_id, ea_active, ea_verified, ea_verify_token FROM email_alerts WHERE ea_email = %s AND ea_lat = %s AND ea_lon = %s",
                    (email, lat, lon), fetch_one=True
                )
                if existing:
                    ea_id, ea_active, ea_verified, old_verify = existing
                    if ea_active and ea_verified:
                        return None  # Already fully active
                    # Reset: new tokens, unverified, inactive
                    self._safe_execute(
                        """UPDATE email_alerts
                           SET ea_active = FALSE, ea_verified = FALSE,
                               ea_verify_token = %s, ea_unsubscribe_token = %s,
                               ea_radius_km = %s, ea_last_notified = NULL
                           WHERE ea_id = %s""",
                        (verify_token, unsub_token, radius_km, ea_id)
                    )
                    self.mydb.commit()
                    return {'ea_id': ea_id, 'verify_token': verify_token, 'unsub_token': unsub_token}

                # New subscription — inactive until verified
                self._safe_execute(
                    """INSERT INTO email_alerts
                       (ea_email, ea_lat, ea_lon, ea_radius_km, ea_active, ea_verified, ea_verify_token, ea_unsubscribe_token)
                       VALUES (%s, %s, %s, %s, FALSE, FALSE, %s, %s)""",
                    (email, lat, lon, radius_km, verify_token, unsub_token)
                )
                self.mydb.commit()
                return {'ea_id': self.mycursor.lastrowid, 'verify_token': verify_token, 'unsub_token': unsub_token}
        except Exception as e:
            print(f"Error adding email alert: {e}")
            return None

    def verify_alert_by_token(self, token):
        """Verify and activate an alert subscription via magic link.

        Returns:
            dict with alert details on success, None if token invalid/already verified
        """
        try:
            with self._lock:
                row = self._safe_fetch(
                    "SELECT ea_id, ea_email, ea_lat, ea_lon, ea_radius_km FROM email_alerts WHERE ea_verify_token = %s",
                    (token,), fetch_one=True
                )
                if not row:
                    return None
                ea_id, ea_email, ea_lat, ea_lon, ea_radius_km = row
                self._safe_execute(
                    "UPDATE email_alerts SET ea_active = TRUE, ea_verified = TRUE WHERE ea_id = %s",
                    (ea_id,)
                )
                self.mydb.commit()
                return {'ea_id': ea_id, 'email': ea_email, 'lat': float(ea_lat), 'lon': float(ea_lon), 'radius_km': ea_radius_km}
        except Exception as e:
            print(f"Error verifying alert: {e}")
            return None

    def get_active_alerts_in_bbox(self, lat_min, lat_max, lon_min, lon_max):
        """Get active+verified alert subscriptions within a bounding box."""
        sql = """
        SELECT ea_id, ea_email, ea_lat, ea_lon, ea_radius_km, ea_last_notified, ea_unsubscribe_token
        FROM email_alerts
        WHERE ea_active = TRUE AND ea_verified = TRUE
          AND ea_lat BETWEEN %s AND %s
          AND ea_lon BETWEEN %s AND %s
        """
        try:
            return self._safe_fetch(sql, (lat_min, lat_max, lon_min, lon_max))
        except Exception as e:
            print(f"Error getting alerts in bbox: {e}")
            return []

    def update_last_notified(self, ea_id):
        """Update the last notified timestamp for an alert subscription."""
        try:
            self._safe_execute("UPDATE email_alerts SET ea_last_notified = NOW() WHERE ea_id = %s", (ea_id,))
            self.mydb.commit()
        except Exception as e:
            print(f"Error updating last_notified: {e}")

    def deactivate_alert_by_token(self, token):
        """Deactivate an alert subscription by its unsubscribe token.

        Works for both active and pending-verification subscriptions.

        Returns:
            True if a subscription was deactivated, False otherwise
        """
        try:
            with self._lock:
                self._safe_execute(
                    "UPDATE email_alerts SET ea_active = FALSE, ea_verified = FALSE WHERE ea_unsubscribe_token = %s AND (ea_active = TRUE OR ea_verified = FALSE)",
                    (token,)
                )
                self.mydb.commit()
                return self.mycursor.rowcount > 0
        except Exception as e:
            print(f"Error deactivating alert: {e}")
            return False

    def get_alerts_by_email(self, email):
        """Get all alert subscriptions for an email address (active or pending verification).

        Returns:
            List of dicts with subscription details
        """
        sql = """
        SELECT ea_id, ea_lat, ea_lon, ea_radius_km, ea_unsubscribe_token, ea_created_at, ea_active, ea_verified
        FROM email_alerts
        WHERE ea_email = %s AND (ea_active = TRUE OR ea_verified = FALSE)
        ORDER BY ea_created_at DESC
        """
        try:
            rows = self._safe_fetch(sql, (email,))
            return [{
                'id': r[0],
                'lat': float(r[1]),
                'lon': float(r[2]),
                'radius_km': r[3],
                'token': r[4],
                'created_at': r[5].isoformat() if r[5] else None,
                'active': bool(r[6]),
                'verified': bool(r[7]),
            } for r in rows]
        except Exception as e:
            print(f"Error getting alerts by email: {e}")
            return []

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
            with self._lock:
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
            return self._safe_fetch(sql, (older_than_hours,))
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
            return self._safe_fetch(sql, (start_time, end_time, min_mag))
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
            row = self._safe_fetch(count_sql, tuple(params), fetch_one=True)
            total = row[0] if row else 0
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
            rows = self._safe_fetch(sql, tuple(params + [limit, offset]))
            columns = ['id', 'prediction_time', 'predicted_lat', 'predicted_lon', 'predicted_dt', 'predicted_mag', 'predicted_place',
                      'actual_lat', 'actual_lon', 'actual_dt', 'actual_mag', 'diff_lat', 'diff_lon', 'diff_dt', 'diff_mag',
                      'verified', 'correct', 'actual_time', 'actual_place']
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
            row = self._safe_fetch(sql, (min_mag * 10,), fetch_one=True)
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
            rows = self._safe_fetch(sql, (min_mag, limit))
            columns = ['id', 'time', 'lat', 'lon', 'mag', 'depth', 'place']
            result = []
            for row in rows:
                d = dict(zip(columns, row))
                if d['time']:
                    # Add Z suffix for UTC timezone consistency with frontend
                    d['time'] = d['time'].isoformat() + 'Z'
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
            row = self._safe_fetch(sql, (pr_id,), fetch_one=True)

            if not row:
                print(f"Prediction {pr_id} not found")
                return False

            pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted = row

            # Convert earthquake coords to encoded format (for storage consistency)
            eq_lat_encoded = int(earthquake_lat + 90)
            eq_lon_encoded = int(earthquake_lon + 180)
            eq_mag_encoded = int(earthquake_mag * 10)

            # Calculate differences
            diff_lat = abs(pr_lat_predicted - eq_lat_encoded) if pr_lat_predicted is not None else None
            diff_lon = abs(pr_lon_predicted - eq_lon_encoded) if pr_lon_predicted is not None else None
            if diff_lon is not None:
                diff_lon = min(diff_lon, 360 - diff_lon)

            # Parse earthquake time and calculate dt
            from datetime import datetime
            if isinstance(earthquake_time, str):
                eq_time = datetime.fromisoformat(earthquake_time.replace('Z', '+00:00'))
            else:
                eq_time = earthquake_time

            actual_dt = int((eq_time.replace(tzinfo=None) - pr_timestamp).total_seconds() / 60)
            diff_dt = abs(pr_dt_predicted - actual_dt) if pr_dt_predicted is not None else None

            diff_mag = abs((pr_mag_predicted / 10.0) - earthquake_mag) if pr_mag_predicted is not None else None

            # Match is correct if within 250km
            correct = distance <= 250

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
            row = self._safe_fetch(sql, (min_mag * 10,), fetch_one=True)
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

            cursor.execute("CALL get_data(%s)", (min_mag,))
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

    def _fetch_emsc(self, days=1, min_magnitude=2):
        """Fetch earthquake data from EMSC (European-Mediterranean Seismological Centre).

        EMSC often has more real-time data than USGS with less delay.
        API: https://www.seismicportal.eu/fdsnws/event/1/query

        Args:
            days: Number of days to fetch
            min_magnitude: Minimum magnitude filter

        Returns:
            List of tuples matching usgs_tmp table format
        """
        from datetime import datetime, timedelta

        all_values = []

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        # EMSC API endpoint (FDSN standard, similar to USGS)
        url = (
            f"https://www.seismicportal.eu/fdsnws/event/1/query"
            f"?format=json"
            f"&starttime={start_time.strftime('%Y-%m-%dT%H:%M:%S')}"
            f"&endtime={end_time.strftime('%Y-%m-%dT%H:%M:%S')}"
            f"&minmagnitude={min_magnitude}"
            f"&limit=10000"
            f"&orderby=time"
        )

        try:
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                print(f"[EMSC] API returned status {response.status_code}")
                return []

            dataset = response.json()
            features = dataset.get('features', [])
            print(f"[EMSC] Fetched {len(features)} earthquakes for last {days} day(s)")

            for feature in features:
                props = feature.get('properties', {})
                geom = feature.get('geometry', {})
                coords = geom.get('coordinates', [0, 0, 0])

                mag = props.get('mag') or 0

                # Use flynn_region as place (EMSC's location description)
                plc = props.get('flynn_region', '')
                if plc:
                    plc = plc.replace("'", "").replace('"', "")

                # Parse time - EMSC uses ISO format
                time_str = props.get('time', '')
                if time_str:
                    try:
                        eq_time = pd.to_datetime(time_str)
                        dtm = eq_time.strftime('%y/%m/%d %H:%M:%S')
                        dat = eq_time.strftime('%y/%m/%d')
                        tim = eq_time.strftime('%H:%M:%S')
                    except:
                        continue
                else:
                    continue

                # Event type - EMSC uses 'evtype' field
                # 'ke' = known earthquake, 'se' = suspected earthquake
                evtype = props.get('evtype', '')
                typ = 'earthquake' if evtype in ['ke', 'se', ''] else evtype

                magType = props.get('magtype', 'm')

                lon = coords[0] if len(coords) > 0 else 0
                lat = coords[1] if len(coords) > 1 else 0
                dep = abs(coords[2]) if len(coords) > 2 else 0  # EMSC sometimes has negative depth

                # Generate unique code based on datetime + location + magnitude (integers only)
                # Format: YYMMDDHHMMSS_LAT_LON_M (short and unique)
                code = f"{eq_time.strftime('%y%m%d%H%M%S')}_{int(lat)}_{int(lon)}_{int(mag)}"

                # Encode for model
                x = lat + 90
                y = lon + 180
                d = dep / 10
                m = mag * 10

                all_values.append((code, dat, tim, dtm, lat, lon, dep, mag, plc, typ, magType, x, y, d, m))

        except requests.exceptions.Timeout:
            print(f"[EMSC] Request timeout")
        except requests.exceptions.RequestException as e:
            print(f"[EMSC] Request error: {e}")
        except Exception as e:
            print(f"[EMSC] Error parsing data: {e}")

        return all_values

    def usgs2DB(self, days=1):
        """Fetch earthquake data from EMSC (primary source - faster, more real-time).

        EMSC (European-Mediterranean Seismological Centre) provides near real-time
        earthquake data with less delay than USGS (~10-15 min vs ~1.5-2 hours).
        """
        self._safe_execute("delete from usgs_tmp")

        min_magnitude = 2
        all_values = []

        # ==================== EMSC API (primary source) ====================
        try:
            all_values = self._fetch_emsc(days=days, min_magnitude=min_magnitude)
        except Exception as e:
            print(f"[EMSC] Error fetching data: {e}")

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

        # Deduplicate: EMSC aggregates reports from multiple agencies (AFAD, KOERI, etc.)
        # so the same earthquake often appears twice with slightly different parameters.
        # Remove the later record when two events are within 10 sec, 0.1°, and 0.3 mag.
        try:
            self._safe_execute("DROP TABLE IF EXISTS _tmp_dedup")
            self._safe_execute("""
                CREATE TABLE _tmp_dedup AS
                SELECT b.us_id as dup_id
                FROM usgs a
                INNER JOIN usgs b
                    ON b.us_id > a.us_id
                    AND b.us_datetime BETWEEN a.us_datetime AND DATE_ADD(a.us_datetime, INTERVAL 10 SECOND)
                    AND ABS(b.us_lat - a.us_lat) < 0.1
                    AND ABS(b.us_lon - a.us_lon) < 0.1
                    AND ABS(b.us_m - a.us_m) <= 3
                WHERE a.us_datetime > DATE_SUB(NOW(), INTERVAL 2 DAY)
            """)
            self.mydb.commit()
            count_result = self._safe_fetch("SELECT COUNT(*) FROM _tmp_dedup", fetch_one=True)
            dedup_count = count_result[0] if count_result else 0
            if dedup_count > 0:
                self._safe_execute("DELETE FROM usgs WHERE us_id IN (SELECT dup_id FROM _tmp_dedup)")
                self.mydb.commit()
                print(f"Dedup: removed {dedup_count} duplicate reports")
            self._safe_execute("DROP TABLE IF EXISTS _tmp_dedup")
            self.mydb.commit()
        except Exception as e:
            print(f"Error dedup: {e}")

        # Calculate us_t = time since last global M4+ earthquake for new records
        # Uses temp table to work around MySQL "can't update table used in FROM clause"
        try:
            count_result = self._safe_fetch("SELECT COUNT(*) FROM usgs WHERE us_t = 0 OR us_t IS NULL", fetch_one=True)
            count = count_result[0] if count_result else 0

            if count > 0:
                print(f"Calculating global M4+ dt for {count} new records...")

                # Step 1: Compute dt values into a temp table
                self._safe_execute("DROP TABLE IF EXISTS _tmp_dt")
                self._safe_execute("""
                    CREATE TABLE _tmp_dt AS
                    SELECT u.us_id,
                        GREATEST(1, COALESCE(
                            (SELECT LEAST(TIMESTAMPDIFF(MINUTE, p.us_datetime, u.us_datetime), 360)
                             FROM usgs p
                             WHERE p.us_datetime < u.us_datetime
                               AND p.us_mag >= 4.0
                             ORDER BY p.us_datetime DESC
                             LIMIT 1),
                            360
                        )) as dt_val
                    FROM usgs u
                    WHERE u.us_t = 0 OR u.us_t IS NULL
                """)
                self.mydb.commit()

                # Step 2: Join-update from temp table
                self._safe_execute("UPDATE usgs u JOIN _tmp_dt t ON u.us_id = t.us_id SET u.us_t = t.dt_val")
                self.mydb.commit()

                self._safe_execute("DROP TABLE IF EXISTS _tmp_dt")
                self.mydb.commit()
                print(f"Done calculating global M4+ dt for {count} records")
        except Exception as e:
            print(f"Error batch calc_dt: {e}")
            import traceback
            traceback.print_exc()

        # Calculate us_lt = time since last M4+ earthquake within 1000km for new records
        try:
            count_result = self._safe_fetch("SELECT COUNT(*) FROM usgs WHERE us_lt IS NULL OR us_lt = 0", fetch_one=True)
            count = count_result[0] if count_result else 0

            if count > 0:
                print(f"Calculating local M4+ dt (1000km) for {count} new records...")

                # For small batches of new records, use SQL with bounding box + haversine
                # Bounding box: ~9° lat ≈ 1000km; lon box = 9 / cos(lat) but we use 36° to be safe up to ~75°
                self._safe_execute("DROP TABLE IF EXISTS _tmp_lt")
                self._safe_execute("""
                    CREATE TABLE _tmp_lt AS
                    SELECT u.us_id,
                        GREATEST(1, COALESCE(
                            (SELECT TIMESTAMPDIFF(MINUTE, p.us_datetime, u.us_datetime)
                             FROM usgs p
                             WHERE p.us_datetime < u.us_datetime
                               AND p.us_mag >= 4.0
                               AND p.us_x BETWEEN u.us_x - 9 AND u.us_x + 9
                               AND p.us_y BETWEEN u.us_y - 36 AND u.us_y + 36
                               AND (6371 * 2 * ASIN(SQRT(
                                   POW(SIN(RADIANS((p.us_x - 90) - (u.us_x - 90)) / 2), 2) +
                                   COS(RADIANS(u.us_x - 90)) * COS(RADIANS(p.us_x - 90)) *
                                   POW(SIN(RADIANS((p.us_y - 180) - (u.us_y - 180)) / 2), 2)
                               ))) < 1000
                             ORDER BY p.us_datetime DESC
                             LIMIT 1),
                            29000000
                        )) as lt_val
                    FROM usgs u
                    WHERE u.us_lt IS NULL OR u.us_lt = 0
                """)
                self.mydb.commit()

                self._safe_execute("UPDATE usgs u JOIN _tmp_lt t ON u.us_id = t.us_id SET u.us_lt = t.lt_val")
                self.mydb.commit()

                self._safe_execute("DROP TABLE IF EXISTS _tmp_lt")
                self.mydb.commit()
                print(f"Done calculating local M4+ dt for {count} records")
        except Exception as e:
            print(f"Error batch calc_lt: {e}")
            import traceback
            traceback.print_exc()

        # Calculate moon features for new records
        try:
            count_result = self._safe_fetch(
                "SELECT COUNT(*) FROM usgs WHERE us_moon_phase = 0 AND us_moon_dist = 0 AND us_datetime IS NOT NULL",
                fetch_one=True
            )
            count = count_result[0] if count_result else 0

            if count > 0:
                print(f"Computing moon features for {count} new records...")
                rows = self._safe_fetch(
                    "SELECT us_id, UNIX_TIMESTAMP(us_datetime) FROM usgs "
                    "WHERE us_moon_phase = 0 AND us_moon_dist = 0 AND us_datetime IS NOT NULL"
                )
                if rows:
                    ids = np.array([r[0] for r in rows], dtype=np.int64)
                    ts = np.array([r[1] for r in rows], dtype=np.float64)
                    moon_phase, moon_dist = self._compute_moon_features(ts)

                    # Only update records where computed value isn't (0,0)
                    needs_update = ~((moon_phase == 0) & (moon_dist == 0))
                    if needs_update.any():
                        batch_data = [
                            (int(moon_phase[j]), int(moon_dist[j]), int(ids[j]))
                            for j in range(len(ids)) if needs_update[j]
                        ]
                        BATCH = 5000
                        for i in range(0, len(batch_data), BATCH):
                            self._safe_execute(
                                "UPDATE usgs SET us_moon_phase = %s, us_moon_dist = %s WHERE us_id = %s",
                                batch_data[i:i+BATCH], many=True
                            )
                            self.mydb.commit()
                    print(f"Done computing moon features for {count} records")
        except Exception as e:
            print(f"Error computing moon features: {e}")

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
            cursor.execute("CALL get_data_fast(%s)", (min_mag,))
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
                nan_count = np.isnan(self.data).sum()
                if nan_count > 0:
                    print(f"  Warning: {nan_count} NaN values replaced with 0 (valid embedding index)")
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
        """Update train/valid splits from loaded data.

        NOTE: Embedding sizes (yr_max, mt_max, etc.) are FIXED constants defined in __init__
        to ensure model architecture consistency. Do NOT update them from data.
        Data values exceeding these limits are clamped in getBatchHybrid/getLast.
        """
        n = int(self.n * len(self.data))
        self.train = self.data[:n]
        self.valid = self.data[n:]

        # Log data statistics for debugging (but don't update max values)
        actual_yr_max = int(self.data[:, 0].max())
        actual_mt_max = int(self.data[:, 1].max())
        actual_x_max = int(self.data[:, 2].max())
        actual_y_max = int(self.data[:, 3].max())
        actual_m_max = int(self.data[:, 4].max())
        actual_d_max = int(self.data[:, 5].max())
        actual_t_max = int(self.data[:, 6].max())

        # Warn if data exceeds model limits (values will be clamped)
        if actual_d_max > self.d_max:
            print(f"  Warning: depth max {actual_d_max} exceeds model limit {self.d_max}, will be clamped")
        if actual_t_max > self.t_raw_max:
            print(f"  Warning: dt max {actual_t_max} exceeds raw cap {self.t_raw_max}, will be clamped")

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

            cursor.execute("CALL get_data_hybrid(%s, %s)", (input_mag, target_mag))
            rows = cursor.fetchall()

            try:
                while cursor.nextset():
                    pass
            except:
                pass

            cursor.close()
            db.close()

            if rows:
                # Stored proc returns 13 columns directly:
                # [yr, mo, x, y, m, d, dt, lt, hour, doy, moon_phase, moon_dist, is_target]
                self.data = np.array(rows, dtype=np.float64)
                nan_count = np.isnan(self.data).sum()
                if nan_count > 0:
                    print(f"  Warning: {nan_count} NaN values replaced with 0 (valid embedding index)")
                    self.data = np.nan_to_num(self.data, 0)

                # Store target indices for efficient sampling
                # Column 12 = is_target
                self.target_indices = np.where(self.data[:, 12] == 1)[0]
                target_count = len(self.target_indices)

                print(f"Loaded {len(self.data):,} records (M{input_mag}+)")
                print(f"  Training targets (M{target_mag}+): {target_count:,} ({100*target_count/len(self.data):.1f}%)")
                print(f"  Earth-state features: hour[0-23], doy[0-365], moon_phase[0-29], moon_dist[0-9]")

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
            raise ValueError("No data loaded — call getData/getDataHybrid first")
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
            't_size' : self.t_max,
            'lt_size' : self.lt_max,
            'hr_size' : self.hr_max,
            'doy_size' : self.doy_max,
            'mp_size' : self.mp_max,
            'md_size' : self.md_max
        }
        return sizes    

    def getScaledData(self):
        """Load data and return it. Note: max values are FIXED constants, not updated from data."""
        data = self.getData()
        # Data statistics (for reference, NOT used for embedding sizes):
        # year [0 55] Some years are active then others
        # month [1 12] some months are active than other, July is the most active month
        # lat [0 180] most activity around 130 degree (encoded: lat+90)
        # lon [0 360] most activity around 70 degree (encoded: lon+180)
        # mag [0 91] encoded as mag*10; 2,3,4 dominate
        # dep [0 75] clamped; 1 and 10 km dominates
        # dt  [0 150] clamped; in minutes
        # NOTE: Embedding sizes are FIXED in __init__ - do NOT update them here
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

        # Extract each target column from the next position — lat, lon, mag only
        targets = {
            'lat': torch.clamp(next_pos[:, 2], 0, 180).long(),  # latitude (0-180)
            'lon': torch.clamp(next_pos[:, 3], 0, 360).long(),  # longitude (0-360)
            'mag': torch.clamp(next_pos[:, 4], 0, 91).long(),   # magnitude (0-91)
        }

        return x.long(), targets

    def getBatchHybrid(self, B, T, split):
        """Get batch for hybrid training with multi-position targets.

        Input has ALL earthquakes, loss is computed at ALL M4+ positions within
        each sequence (not just the last one). This gives ~100-180x more gradient
        signal per batch compared to single-position training.

        Returns:
            x: Input sequences [B, T, 8] - includes all earthquakes as context
            targets: Dict with 'lat', 'lon', 'dt', 'mag' tensors [B, T] - targets at every position
            target_mask: Boolean tensor [B, T] - True where position predicts an M4+ next event
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
        # Use first 12 columns (exclude is_target column at index 12)
        # Columns: [yr, mo, lat, lon, mag, depth, dt_global, dt_local, hour, doy, moon_phase, moon_dist]
        data_tensor = torch.from_numpy(self.data[:, :12])
        is_target_col = torch.from_numpy(self.data[:, 12])

        x = torch.stack([data_tensor[int(pos)-T:int(pos)] for pos in target_positions])

        # CRITICAL: Clamp input data to model embedding ranges
        # Columns: 0=year, 1=month, 2=lat, 3=lon, 4=mag, 5=depth, 6=dt_global, 7=dt_local,
        #          8=hour, 9=doy, 10=moon_phase, 11=moon_dist
        x[:, :, 0] = torch.clamp(x[:, :, 0], 0, self.yr_max)      # year: 0-60
        x[:, :, 1] = torch.clamp(x[:, :, 1] - 1, 0, 11)           # month: convert 1-12 to 0-11 for embedding
        x[:, :, 2] = torch.clamp(x[:, :, 2], 0, self.x_max)       # lat: 0-180
        x[:, :, 3] = torch.clamp(x[:, :, 3], 0, self.y_max)       # lon: 0-360
        x[:, :, 4] = torch.clamp(x[:, :, 4], 0, self.m_max)       # mag: 0-91
        x[:, :, 5] = torch.clamp(x[:, :, 5], 0, self.d_max)       # depth: 0-200
        x[:, :, 6] = self._log_bin_tensor(torch.clamp(x[:, :, 6], 0, self.t_raw_max))            # global dt: log-binned 0-9
        x[:, :, 7] = self._log_bin_tensor(torch.clamp(x[:, :, 7], 0, self.lt_raw_max), n_bins=26)  # local dt: log-binned 0-25
        x[:, :, 8] = torch.clamp(x[:, :, 8], 0, self.hr_max)      # hour: 0-23
        x[:, :, 9] = torch.clamp(x[:, :, 9], 0, self.doy_max)     # day of year: 0-365
        x[:, :, 10] = torch.clamp(x[:, :, 10], 0, self.mp_max)    # moon phase: 0-29
        x[:, :, 11] = torch.clamp(x[:, :, 11], 0, self.md_max)    # moon distance: 0-9

        # Multi-position targets: for each position j, the target is position j+1
        # "next positions" = data[pos-T+1 : pos+1] — shifted by 1 from input
        next_data = torch.stack([data_tensor[int(pos)-T+1:int(pos)+1] for pos in target_positions])  # [B, T, 12]
        next_is_target = torch.stack([is_target_col[int(pos)-T+1:int(pos)+1] for pos in target_positions])  # [B, T]

        # Target mask: True where the NEXT position is M4+ (loss computed there)
        # Position T-1 always True (anchor target is M4+ by construction)
        target_mask = (next_is_target == 1)  # [B, T] boolean

        # Build multi-position targets [B, T] — lat, lon, mag only (no dt prediction)
        targets = {
            'lat': torch.clamp(next_data[:, :, 2], 0, 180).long(),
            'lon': torch.clamp(next_data[:, :, 3], 0, 360).long(),
            'mag': torch.clamp(next_data[:, :, 4], 0, 91).long(),
        }

        return x.long(), targets, target_mask

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

        # CRITICAL: Clamp input data to model embedding ranges
        # Column indices: 0=year, 1=month, 2=lat, 3=lon, 4=mag, 5=depth, 6=dt
        x[:, :, 0] = torch.clamp(x[:, :, 0], 0, self.yr_max)      # year: 0-60
        x[:, :, 1] = torch.clamp(x[:, :, 1] - 1, 0, 11)           # month: convert 1-12 to 0-11 for embedding
        x[:, :, 2] = torch.clamp(x[:, :, 2], 0, self.x_max)       # lat: 0-180
        x[:, :, 3] = torch.clamp(x[:, :, 3], 0, self.y_max)       # lon: 0-360
        x[:, :, 4] = torch.clamp(x[:, :, 4], 0, self.m_max)       # mag: 0-91
        x[:, :, 5] = torch.clamp(x[:, :, 5], 0, self.d_max)       # depth: 0-200
        x[:, :, 6] = self._log_bin_tensor(torch.clamp(x[:, :, 6], 0, self.t_raw_max))  # dt: log-binned 0-9

        # For y, return the last record's target column (used for reference only)
        y = data_[-1, col].unsqueeze(0)  # [1]

        return x.long(), y.long()
    

    def getLastFromDB(self, T, input_mag=2.0):
        """Get the LAST T earthquakes directly from database for FRESH predictions.

        This method queries the database directly instead of using the cached
        data array, ensuring predictions use the most recent earthquake data.

        Args:
            T: Sequence length (number of recent earthquakes to fetch)
            input_mag: Minimum magnitude filter (default 2.0)

        Returns:
            x: Last T earthquakes [1, T, 12] as tensor ready for model input
        """
        try:
            sql = """
            SELECT year, month, us_x, us_y, us_m, us_d, us_t, us_lt,
                   hour_of_day, day_of_year, us_moon_phase, us_moon_dist
            FROM (
                SELECT
                    YEAR(us_datetime) - 1970 as year,
                    MONTH(us_datetime) as month,
                    us_x,
                    us_y,
                    us_m,
                    GREATEST(us_d, 0) as us_d,
                    CASE
                        WHEN us_t IS NULL OR us_t = 0 THEN 360
                        WHEN us_t > 360 THEN 360
                        ELSE us_t
                    END as us_t,
                    CASE
                        WHEN us_lt IS NULL OR us_lt = 0 THEN 29000000
                        ELSE us_lt
                    END as us_lt,
                    HOUR(us_datetime) as hour_of_day,
                    DAYOFYEAR(us_datetime) - 1 as day_of_year,
                    us_moon_phase,
                    us_moon_dist,
                    us_datetime
                FROM usgs
                WHERE us_mag >= %s
                  AND us_type = 'earthquake'
                  AND us_magtype LIKE 'm%%'
                ORDER BY us_datetime DESC
                LIMIT %s
            ) sub
            ORDER BY us_datetime ASC
            """

            rows = self._safe_fetch(sql, (input_mag, T))

            if not rows or len(rows) < T:
                print(f"Warning: Only got {len(rows) if rows else 0} earthquakes, need {T}")
                return None

            # 12 columns: [yr, mo, x, y, m, d, dt, lt, hour, doy, moon_phase, moon_dist]
            data = np.array(rows, dtype=np.float64)
            data = np.nan_to_num(data, 0)

            x = torch.from_numpy(data).unsqueeze(0)  # [1, T, 12]

            # CRITICAL: Clamp input data to model embedding ranges
            # Columns: 0=year, 1=month, 2=lat, 3=lon, 4=mag, 5=depth, 6=dt_global, 7=dt_local,
            #          8=hour, 9=doy, 10=moon_phase, 11=moon_dist
            x[:, :, 0] = torch.clamp(x[:, :, 0], 0, self.yr_max)      # year: 0-60
            x[:, :, 1] = torch.clamp(x[:, :, 1] - 1, 0, 11)           # month: convert 1-12 to 0-11 for embedding
            x[:, :, 2] = torch.clamp(x[:, :, 2], 0, self.x_max)       # lat: 0-180
            x[:, :, 3] = torch.clamp(x[:, :, 3], 0, self.y_max)       # lon: 0-360
            x[:, :, 4] = torch.clamp(x[:, :, 4], 0, self.m_max)       # mag: 0-91
            x[:, :, 5] = torch.clamp(x[:, :, 5], 0, self.d_max)       # depth: 0-200
            x[:, :, 6] = self._log_bin_tensor(torch.clamp(x[:, :, 6], 0, self.t_raw_max))            # global dt: log-binned 0-9
            x[:, :, 7] = self._log_bin_tensor(torch.clamp(x[:, :, 7], 0, self.lt_raw_max), n_bins=26)  # local dt: log-binned 0-25
            x[:, :, 8] = torch.clamp(x[:, :, 8], 0, self.hr_max)      # hour: 0-23
            x[:, :, 9] = torch.clamp(x[:, :, 9], 0, self.doy_max)     # day of year: 0-365
            x[:, :, 10] = torch.clamp(x[:, :, 10], 0, self.mp_max)    # moon phase: 0-29
            x[:, :, 11] = torch.clamp(x[:, :, 11], 0, self.md_max)    # moon distance: 0-9

            return x.long()

        except Exception as e:
            print(f"Error in getLastFromDB: {e}")
            import traceback
            traceback.print_exc()
            return None

    def closeAll(self):
        """Close the database connection."""
        self._close_connection()
        