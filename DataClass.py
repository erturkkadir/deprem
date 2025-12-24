import requests
import pandas as pd
import mysql.connector
import datetime
import csv
import numpy as np
import torch
import config


class DataC():
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

        self._connect_db()
        self.fname =  "./data/latest.csv"

        # Ensure predictions table exists
        self._create_predictions_table()

    def _connect_db(self):
        """Connect or reconnect to database"""
        try:
            self.mydb = mysql.connector.connect(
                host=config.DB_HOST,
                user=config.DB_USER,
                password=config.DB_PASS,
                database=config.DB_NAME,
                autocommit=True,
                connection_timeout=30,
                ssl_disabled=True
            )
            self.mycursor = self.mydb.cursor()
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

    def _ensure_connection(self):
        """Ensure database connection is active, reconnect if needed"""
        try:
            if self.mydb is None or not self.mydb.is_connected():
                self._connect_db()
            else:
                self.mydb.ping(reconnect=True, attempts=3, delay=1)
                # Refresh cursor after ping
                self.mycursor = self.mydb.cursor()
        except Exception as e:
            print(f"Reconnecting to database: {e}")
            self._connect_db()

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
            self.mycursor.execute(sql)
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
                self.mycursor.execute(sql)
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
        self._ensure_connection()
        sql = """
        INSERT INTO predictions (pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted, pr_place)
        VALUES (NOW(), %s, %s, %s, %s, %s)
        """
        try:
            self.mycursor.execute(sql, (lat_predicted, lon_predicted, dt_predicted, mag_predicted, place))
            self.mydb.commit()
            return self.mycursor.lastrowid
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return None

    def get_unverified_predictions(self, older_than_hours=24):
        """Get predictions that haven't been verified yet"""
        self._ensure_connection()
        sql = """
        SELECT pr_id, pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted
        FROM predictions
        WHERE pr_verified = FALSE
        AND pr_timestamp < DATE_SUB(NOW(), INTERVAL %s HOUR)
        ORDER BY pr_timestamp ASC
        """
        try:
            self.mycursor.execute(sql, (older_than_hours,))
            return self.mycursor.fetchall()
        except Exception as e:
            print(f"Error getting unverified predictions: {e}")
            return []

    def get_earthquakes_in_window(self, start_time, end_time, min_mag=4.0):
        """Get actual earthquakes within a time window"""
        self._ensure_connection()
        sql = """
        SELECT us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place
        FROM usgs
        WHERE us_datetime BETWEEN %s AND %s
        AND us_mag >= %s
        AND us_type = 'earthquake'
        ORDER BY us_datetime ASC
        """
        try:
            self.mycursor.execute(sql, (start_time, end_time, min_mag))
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
        self._ensure_connection()
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
            self.mycursor.execute(sql, (actual_id, actual_lat, actual_lon, actual_dt, actual_mag, actual_time, diff_lat, diff_lon, diff_dt, diff_mag, correct, pr_id))
            self.mydb.commit()
            return True
        except Exception as e:
            print(f"Error verifying prediction: {e}")
            return False

    def get_predictions_with_actuals(self, limit=50, min_mag=4.0):
        """Get predictions with matched actual earthquakes for display

        Args:
            limit: Maximum number of predictions to return
            min_mag: Minimum predicted magnitude to show (default 4.0)
        """
        self._ensure_connection()
        sql = """
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
        WHERE p.pr_mag_predicted >= %s
        ORDER BY p.pr_timestamp DESC
        LIMIT %s
        """
        try:
            # min_mag stored as encoded (mag * 10), so multiply threshold
            self.mycursor.execute(sql, (min_mag * 10, limit,))
            columns = ['id', 'prediction_time', 'predicted_lat', 'predicted_lon', 'predicted_dt', 'predicted_mag', 'predicted_place',
                      'actual_lat', 'actual_lon', 'actual_dt', 'actual_mag', 'diff_lat', 'diff_lon', 'diff_dt', 'diff_mag',
                      'verified', 'correct', 'actual_time', 'actual_place']
            rows = self.mycursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"Error getting predictions with actuals: {e}")
            return []

    def get_latest_prediction(self, min_mag=4.0):
        """Get the most recent prediction with magnitude >= min_mag"""
        self._ensure_connection()
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
            self.mycursor.execute(sql, (min_mag * 10,))
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
        self._ensure_connection()
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
            self.mycursor.execute(sql, (min_mag, limit))
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
        self._ensure_connection()

        try:
            # First get the prediction data to calculate differences
            sql = """SELECT pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted
                     FROM predictions WHERE pr_id = %s"""
            self.mycursor.execute(sql, (pr_id,))
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
            self.mycursor.execute(sql, (
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
        self._ensure_connection()
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
            self.mycursor.execute(sql, (min_mag * 10,))
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
        # Create fresh connection for export (avoids cursor state issues)
        self._connect_db()

        sql = f"CALL get_data({min_mag})"
        self.mycursor.execute(sql)
        rows = self.mycursor.fetchall()

        # Consume any remaining results from stored procedure
        while self.mycursor.nextset():
            pass

        cols = [i[0] for i in self.mycursor.description] if self.mycursor.description else []

        if rows and cols:
            fp = open(self.fname, 'w')
            myFile = csv.writer(fp, lineterminator = '\n')
            myFile.writerow(cols)
            myFile.writerows(rows)
            fp.close()
            print(f"Exported {len(rows)} records to CSV")

        self.mydb.close()
        
    def usgs2DB(self):
        self._ensure_connection()
        self.mycursor.execute("delete from usgs_tmp")

        min_magnitude = 2
        all_values = []

        # Collect all data from USGS API first
        for i in range(3, 0, -1):
            start_time = f"now-{i}days"
            end_time = f"now-{(i-1)}days"
            path = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_time}&endtime={end_time}&minmagnitude={min_magnitude}"

            try:
                response = requests.get(path, timeout=30)
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
            self._ensure_connection()
            sql = """INSERT INTO usgs_tmp(us_code, us_date, us_time, us_datetime, us_lat, us_lon,
                     us_dep, us_mag, us_place, us_type, us_magType, us_x, us_y, us_d, us_m)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            try:
                self.mycursor.executemany(sql, all_values)
                self.mydb.commit()
                print(f"Batch inserted {len(all_values)} earthquakes")
            except Exception as e:
                print(f"Batch insert error: {e}")

        # Run stored procedure to merge into main table
        self._ensure_connection()
        try:
            self.mycursor.execute("call ins_quakes()")
            print("Executed ins_quakes()")
        except Exception as e:
            print(f"Error ins_quakes: {e}")

        # Calculate time differences for new records
        self._ensure_connection()
        self.mycursor.execute("SELECT us_id FROM usgs WHERE us_t is null")
        rows = self.mycursor.fetchall()
        if rows:
            print(f"Calculating dt for {len(rows)} new records...")
            for row in rows:
                try:
                    self.mycursor.execute(f"call calc_dt({row[0]})")
                except Exception as e:
                    print(f"Error calc_dt for {row[0]}: {e}")
                    self._ensure_connection()
            self.mydb.commit()
            print("Done calculating dt")
        
    def getData(self):
        
        df1 = pd.read_csv(self.fname)
        df1.fillna(0, inplace=True)
        self.data = np.array(df1.values)
        n = int(self.n*len(self.data))
        self.train = self.data[:n]
        self.valid = self.data[n:]
        # print(f"&& data size  {len(self.data ):,}")
        # print(f"&& train size {len(self.train):,}")
        # print(f"&& valid size {len(self.valid):,}")
        
        self.yr_max = self.data[:, 0].max() + 1  # year 0 to 54
        self.mt_max = self.data[:, 1].max() + 1 
        self.x_max  = self.data[:, 2].max() + 1 
        self.y_max  = self.data[:, 3].max() + 1 
        self.m_max  = self.data[:, 4].max() + 1 
        self.d_max  = self.data[:, 5].max() + 1 
        self.t_max  = self.data[:, 6].max() + 1 
        
        return self.data
        
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
        

    def getLast(self, B, T, split, col):
        
        if (len(self.data)<1):            
            self.getData()

        data_ = self.train if split=='train' else self.valid
                
        data_ = torch.from_numpy(data_)

        ix = torch.randint(len(data_)-T, (B,))
        
        # ix = range(B)
        # print(f"IX IX {ix}")
    

        x = torch.stack([data_[i:i+T] for i in ix])
        y = torch.stack([data_[i+T:i+T+1] for i in ix])
        # print(f"INFERENCE X {x.shape} {y.shape}")


        y = y[-1, :, col]
        # print(f"INFERENCE X {x.shape} {y.shape}")
        # print(f"INFERENCE X {y}")
        # print(f"{1/0}")
        
        return x.long(), y.long()
    

    def closeAll(self):
        self.mydb.close()
        