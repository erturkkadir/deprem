"""
Earthquake Prediction Server
- Loads latest checkpoint from training
- Makes predictions every 5 minutes
- Saves predictions to database
- Auto-verifies predictions against actual earthquakes
- Serves API for frontend on port 3000
"""
import os
import sys
import glob
import math
import torch
import atexit
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from torch.nn import functional as F

from DataClass import DataC
from EqModelComplex import EqModelComplex

# Force unbuffered output for real-time monitoring
sys.stdout.reconfigure(line_buffering=True)


def reverse_geocode(lat, lon):
    """Convert lat/lon to location name using OpenStreetMap Nominatim"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'zoom': 5,
            'addressdetails': 1
        }
        headers = {'User-Agent': 'EarthquakePredictionSystem/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            parts = []

            if address.get('city'):
                parts.append(address['city'])
            elif address.get('town'):
                parts.append(address['town'])
            elif address.get('village'):
                parts.append(address['village'])
            elif address.get('county'):
                parts.append(address['county'])

            if address.get('state'):
                parts.append(address['state'])

            if address.get('country'):
                parts.append(address['country'])

            if parts:
                return ', '.join(parts)

            display = data.get('display_name', '')
            if display:
                parts = display.split(', ')[:3]
                return ', '.join(parts)

        return None
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
        return None


app = Flask(__name__, static_folder='web/dist')
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Model configuration - must match train.py
# Use CPU for server (training uses GPU)
device = "cpu"
col = 2  # Latitude prediction
p_max = 181
B = 1
T = 512           # Sequence length (must match training)
n_embed = 1176    # Embedding size (must match training, divisible by 8 heads)
n_heads = 8
n_layer = 6       # Must match training
dropout = 0.1      # Must match train.py
INPUT_MAG = 2.0   # Min magnitude for input context
TARGET_MAG = 4.0  # Min magnitude for targets

# Global state
model = None
dataC = None
scheduler = None
current_checkpoint = None  # Track which checkpoint is loaded

MODEL_DIR = '/var/www/syshuman/quake'
MODEL_PREFIX = 'eqModel_complex'

# Matching criteria (from README.md)
MATCH_RADIUS_KM = 250       # Haversine distance for matching (km)
MAGNITUDE_TOLERANCE = 0.75  # ±0.75 magnitude
MIN_MAG_DISPLAY = 4.0       # Only show predictions with mag >= 4.0
MIN_PREDICTION_WINDOW = 10  # minutes - reject predictions with window < 10 min
LATE_SEARCH_HOURS = 72      # hours - continue searching after MISSED before closing
EARTH_RADIUS_KM = 6371      # km


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km between two points."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_latest_checkpoint():
    """Find the latest checkpoint file by timestamp"""
    pattern = f'{MODEL_DIR}/{MODEL_PREFIX}_*.pth'
    checkpoints = glob.glob(pattern)

    if checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]

    # Fallback to legacy path
    legacy = f'{MODEL_DIR}/{MODEL_PREFIX}.pth'
    if os.path.exists(legacy):
        return legacy

    return None


def load_model(force_reload=False):
    """Load or reload the earthquake prediction model"""
    global model, dataC, current_checkpoint

    checkpoint_path = get_latest_checkpoint()

    # Skip if same checkpoint already loaded (unless forced)
    if not force_reload and checkpoint_path == current_checkpoint and model is not None:
        return False  # No reload needed

    print(f"[{datetime.now()}] Loading model...")

    dataC = DataC()
    dataC.getDataHybrid(input_mag=INPUT_MAG, target_mag=TARGET_MAG)

    sizes = dataC.getSizes()

    model = EqModelComplex(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max, use_rope=True, use_gpe=True)
    model.to(device)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Clean keys from torch.compile prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value

        try:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            current_checkpoint = checkpoint_path
            print(f"[{datetime.now()}] Loaded checkpoint: {os.path.basename(checkpoint_path)}")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"[{datetime.now()}] Checkpoint incompatible, using random weights")
                print(f"  Error: {e}")
            else:
                raise e
    else:
        print("No checkpoint found, using random weights")

    model.eval()
    return True


def reload_checkpoint_weights():
    """Reload just the checkpoint weights without reloading data"""
    global model, current_checkpoint

    if model is None:
        return False

    checkpoint_path = get_latest_checkpoint()
    if not checkpoint_path:
        return False

    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)
        current_checkpoint = checkpoint_path
        print(f"[{datetime.now()}] Reloaded weights: {os.path.basename(checkpoint_path)}")
        return True
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"[{datetime.now()}] Checkpoint incompatible with current model, skipping")
        else:
            print(f"[{datetime.now()}] Error loading checkpoint: {e}")
        return False


def reload_if_new_checkpoint():
    """Check for new checkpoint and reload if found"""
    global current_checkpoint
    import time

    latest = get_latest_checkpoint()
    if latest and latest != current_checkpoint:
        print(f"[{datetime.now()}] New checkpoint detected: {os.path.basename(latest)}")

        # Wait for file to finish writing (check if file size is stable)
        try:
            size1 = os.path.getsize(latest)
            time.sleep(2)  # Wait 2 seconds
            size2 = os.path.getsize(latest)

            if size1 != size2:
                print(f"[{datetime.now()}] Checkpoint still being written, skipping this cycle")
                return False
        except OSError:
            return False

        # Just reload weights, not the entire data
        return reload_checkpoint_weights()
    return False


def make_prediction():
    """Make a prediction and save to database"""
    global model, dataC

    if model is None or dataC is None:
        print("Model or data not loaded")
        return None

    try:
        # RACE CONDITION PREVENTION: Don't create duplicate predictions within 30 seconds
        existing = dataC.get_latest_prediction(min_mag=4.0)
        if existing and not existing.get('verified'):
            pr_timestamp = datetime.fromisoformat(existing['timestamp'])
            if (datetime.now() - pr_timestamp).total_seconds() < 30:
                print(f"[{datetime.now()}] Skipping - prediction #{existing['id']} created {(datetime.now() - pr_timestamp).total_seconds():.0f}s ago")
                return existing['id']

        # IMPORTANT: Use getLastFromDB() to get FRESH data directly from database
        # This ensures predictions use the most recent earthquakes, not stale cached data
        x_test = dataC.getLastFromDB(T, input_mag=INPUT_MAG)
        if x_test is None:
            print(f"[{datetime.now()}] Error: Could not get fresh data from database")
            return None
        x_test = x_test.to(device)

        with torch.no_grad():
            predictions = model.generate(x_test)

        lat_encoded = predictions['lat']
        lon_encoded = predictions['lon']
        dt_minutes = predictions['dt']
        mag_encoded = predictions['mag']

        # Convert to actual coordinates
        lat_actual = lat_encoded - 90
        lon_actual = lon_encoded - 180
        mag_actual = mag_encoded / 10.0

        # Get location name
        place = reverse_geocode(lat_actual, lon_actual)

        # Save to database
        pr_id = dataC.save_prediction(lat_encoded, lon_encoded, dt_minutes, mag_encoded, place)

        place_str = f" near {place}" if place else ""
        print(f"[{datetime.now()}] Prediction: lat={lat_actual}, lon={lon_actual}{place_str}, dt={dt_minutes}min, mag={mag_actual} (id={pr_id})")
        return pr_id

    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def auto_verify_predictions():
    """72h late search: check missed predictions for late matches.

    Predictions marked as missed (verified=True, correct=False, no actual_id)
    are re-checked for up to 72 hours after their predicted event time.
    If a match is found → LATE CATCH. If 72h passes → MISSED FINAL (stays closed).
    """
    global dataC

    if dataC is None:
        return

    try:
        # Get predictions that are missed but might still get a late catch
        # These are: verified=True, correct=False, actual_id IS NULL, within 72h of predicted event
        sql = """
            SELECT pr_id, pr_timestamp, pr_lat_predicted, pr_lon_predicted,
                   pr_dt_predicted, pr_mag_predicted
            FROM predictions
            WHERE pr_verified = 1 AND pr_correct = 0 AND pr_actual_id IS NULL
              AND pr_timestamp > DATE_SUB(NOW(), INTERVAL %s HOUR)
        """
        missed_preds = dataC._safe_fetch(sql, (LATE_SEARCH_HOURS + 24,))

        for pred in missed_preds:
            pr_id, pr_timestamp, pr_lat, pr_lon, pr_dt, pr_mag = pred

            pred_lat_actual = (pr_lat - 90) if pr_lat else None
            pred_lon_actual = (pr_lon - 180) if pr_lon else None
            pred_mag_actual = (pr_mag / 10.0) if pr_mag else 4.0

            # Search window: from prediction time to prediction_time + dt + 72h
            expected_event_time = pr_timestamp + timedelta(minutes=pr_dt or 60)
            late_search_end = expected_event_time + timedelta(hours=LATE_SEARCH_HOURS)

            # Only search if we're still within the 72h window
            now = datetime.now()
            if now > late_search_end:
                continue  # Already past 72h - stays as MISSED FINAL

            # Match = M4+ earthquake within 250km of predicted location
            actuals = dataC.get_earthquakes_in_window(pr_timestamp, now, min_mag=4.0)

            for actual in (actuals or []):
                us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place = actual

                eq_lat = us_x - 90
                eq_lon = us_y - 180
                dist_km = haversine_km(pred_lat_actual, pred_lon_actual, eq_lat, eq_lon) if pred_lat_actual is not None else None

                if dist_km is not None and dist_km <= MATCH_RADIUS_KM:
                    actual_dt = int((us_datetime - pr_timestamp).total_seconds() / 60)
                    diff_dt = abs((pr_dt or 60) - actual_dt)
                    diff_mag = abs(pred_mag_actual - us_mag) if us_mag else None

                    print(f"[{now}] LATE CATCH for prediction #{pr_id}: M{us_mag} at {dist_km:.0f}km - {us_place}")

                    dataC.verify_prediction(
                        pr_id=pr_id, actual_id=us_id,
                        actual_lat=us_x, actual_lon=us_y,
                        actual_dt=actual_dt,
                        actual_mag=int(us_mag * 10) if us_mag else None,
                        actual_time=us_datetime,
                        diff_lat=int(abs(pr_lat - us_x)) if pr_lat else None,
                        diff_lon=int(abs(pr_lon - us_y)) if pr_lon else None,
                        diff_dt=int(diff_dt),
                        diff_mag=diff_mag, correct=True
                    )
                    break  # First match wins

    except Exception as e:
        print(f"Error in auto-verification: {e}")
        import traceback
        traceback.print_exc()


def check_and_handle_prediction():
    """
    Prediction cycle following README.md spec:

    Status flow: PENDING → MATCHED (green, closed)
                        → MISSED (yellow) → search 72h → LATE CATCH (orange) or MISSED FINAL (red)

    When window expires (MISSED), immediately create a NEW prediction.
    The old prediction continues its 72h late search in the background via auto_verify_predictions().
    """
    global dataC

    if dataC is None:
        return None

    try:
        now = datetime.now()
        latest = dataC.get_latest_prediction(min_mag=4.0)

        # No prediction exists - make one
        if not latest:
            print(f"[{now}] No active prediction, creating new one...")
            make_prediction()
            return {'action': 'new_prediction', 'reason': 'no_prediction'}

        # Prediction already verified/closed - make new one
        if latest.get('verified'):
            print(f"[{now}] Previous prediction closed, creating new one...")
            make_prediction()
            return {'action': 'new_prediction', 'reason': 'previous_closed'}

        # Active PENDING prediction
        pr_id = latest['id']
        pr_timestamp = datetime.fromisoformat(latest['timestamp'])
        pred_lat_actual = latest.get('predicted_lat')   # already decoded by get_latest_prediction()
        pred_lon_actual = latest.get('predicted_lon')   # already decoded
        pred_mag_actual = latest.get('predicted_mag') or 4.0  # already decoded
        pr_dt = latest.get('predicted_dt') or 60

        if pred_lat_actual is None or pred_lon_actual is None:
            return None

        # Prediction window
        expected_event_time = pr_timestamp + timedelta(minutes=pr_dt)

        # Match = first M4+ earthquake that occurred since the prediction
        # Match = M4+ earthquake within 250km of predicted location AND within dt window
        recent_quakes = dataC.get_earthquakes_in_window(pr_timestamp, now, min_mag=4.0)

        if recent_quakes:
            for us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place in recent_quakes:
                eq_lat = us_x - 90
                eq_lon = us_y - 180
                dist_km = haversine_km(pred_lat_actual, pred_lon_actual, eq_lat, eq_lon)

                if dist_km <= MATCH_RADIUS_KM:
                    actual_dt = int((us_datetime - pr_timestamp).total_seconds() / 60)
                    diff_dt = abs(pr_dt - actual_dt) if pr_dt else None
                    diff_mag = abs(pred_mag_actual - us_mag) if us_mag else None

                    print(f"\n{'='*60}")
                    print(f"[{now}] MATCHED! M{us_mag} within {dist_km:.0f}km")
                    print(f"  Prediction #{pr_id}: {pred_lat_actual:.1f}°, {pred_lon_actual:.1f}°, M{pred_mag_actual:.1f}, dt={pr_dt}min")
                    print(f"  Actual: {eq_lat:.1f}°, {eq_lon:.1f}° - M{us_mag} - {us_place}")
                    print(f"{'='*60}\n")

                    dataC.verify_prediction(
                        pr_id=pr_id, actual_id=us_id,
                        actual_lat=us_x, actual_lon=us_y,
                        actual_dt=actual_dt,
                        actual_mag=int(us_mag * 10) if us_mag else None,
                        actual_time=us_datetime,
                        diff_lat=int(abs(pred_lat_actual - eq_lat)),
                        diff_lon=int(abs(pred_lon_actual - eq_lon)),
                        diff_dt=int(diff_dt) if diff_dt else None,
                        diff_mag=diff_mag, correct=True
                    )

                    # Make new prediction immediately
                    print(f"[{now}] Creating new prediction...")
                    make_prediction()
                    return {
                        'action': 'matched',
                        'prediction_id': pr_id,
                        'earthquake_id': us_id,
                        'distance_km': round(dist_km),
                        'place': us_place,
                        'mag': us_mag
                    }

        # No match found - check if window expired
        if now > expected_event_time:
            # Window expired → MISSED status
            # Mark as missed and immediately create new prediction
            # The 72h late search continues via auto_verify_predictions()
            print(f"[{now}] Prediction #{pr_id} window expired (dt={pr_dt}min), status: MISSED")
            print(f"  Late search continues for {LATE_SEARCH_HOURS}h via auto_verify")

            dataC.verify_prediction(
                pr_id=pr_id, actual_id=None,
                actual_lat=None, actual_lon=None, actual_dt=None,
                actual_mag=None, actual_time=None,
                diff_lat=None, diff_lon=None, diff_dt=None,
                diff_mag=None, correct=False
            )

            # Create new prediction right away
            print(f"[{now}] Creating new prediction...")
            make_prediction()
            return {'action': 'missed_new_prediction', 'prediction_id': pr_id}

        # Still PENDING - show countdown
        remaining = (expected_event_time - now).total_seconds() / 60
        return {'action': 'pending', 'prediction_id': pr_id, 'remaining_minutes': round(remaining, 1)}

    except Exception as e:
        print(f"Error in check_and_handle_prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def refresh_data(full_refresh=False):
    """Pull latest USGS data and reload hybrid data

    Args:
        full_refresh: If True, fetch 3 days. If False, only fetch last day (faster)
    """
    global dataC

    try:
        days = 3 if full_refresh else 1
        print(f"[{datetime.now()}] Refreshing USGS data (last {days} day(s))...")

        if dataC is None:
            dataC = DataC()

        dataC.usgs2DB(days=days)

        # Reload hybrid data
        dataC = DataC()
        dataC.getDataHybrid(input_mag=INPUT_MAG, target_mag=TARGET_MAG)

        print(f"[{datetime.now()}] Data refresh complete")
        return True

    except Exception as e:
        print(f"Error refreshing data: {e}")
        # Just reconnect to database, don't reload all data
        try:
            if dataC is not None:
                dataC._connect_db()
        except:
            pass
        return False


def monitor_cycle():
    """
    Main monitoring loop - runs every 120 seconds:
    1. Quick refresh USGS data (last day only)
    2. Check prediction status (match/expire/waiting)
    """
    global dataC
    import time
    start_time = time.time()

    # Run within Flask app context
    with app.app_context():
        try:
            # Check for new model checkpoint
            reload_if_new_checkpoint()

            # Quick USGS data refresh (just update DB)
            if dataC is None:
                dataC = DataC()
            dataC.usgs2DB(days=1)

            # Check and handle current prediction
            result = check_and_handle_prediction()
            if result:
                action = result.get('action')
                print(f"[{datetime.now()}] Status: {action} - {result}")

            # Verify old predictions (>24 hours)
            auto_verify_predictions()

            elapsed = time.time() - start_time
            if elapsed > 60:
                print(f"[{datetime.now()}] Warning: monitor_cycle took {elapsed:.1f}s")

        except Exception as e:
            print(f"Error in monitor cycle: {e}")
            import traceback
            traceback.print_exc()
            # Just reconnect to database, don't reload all data
            try:
                if dataC is not None:
                    dataC._connect_db()
            except:
                pass


def start_scheduler():
    """Start background scheduler"""
    global scheduler

    scheduler = BackgroundScheduler()

    # Single monitoring job every 120 seconds
    # Allow coalesce=True to merge missed runs, misfire_grace_time to handle delays
    scheduler.add_job(
        func=monitor_cycle,
        trigger="interval",
        seconds=120,
        id='monitor_cycle',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60
    )

    scheduler.start()
    print("Scheduler started: monitoring every 120 seconds")

    atexit.register(lambda: scheduler.shutdown())


# ============= API ENDPOINTS =============

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory(os.path.join(app.static_folder, 'assets'), path)


@app.route('/<path:path>')
def serve_static(path):
    # For client-side routing (React Router), serve index.html for non-file paths
    # Check if the path is an actual file in the static folder
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    # Otherwise, serve index.html and let React Router handle the route
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/live', methods=['GET'])
def get_live_data():
    """Get live data: latest prediction, recent earthquakes, stats, and match info.
    Also handles match detection and triggers new prediction when match found or expired."""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        latest_prediction = dataC.get_latest_prediction()
        recent_earthquakes = dataC.get_recent_earthquakes(limit=50, min_mag=2.0)
        stats = dataC.get_prediction_stats()

        # Calculate match info for each earthquake
        match_info = None
        closest_match = None
        matched_earthquake = None
        prediction_status = None  # Will hold computed status info for frontend
        now = datetime.now(timezone.utc).replace(tzinfo=None)  # Use UTC for consistency

        # If the latest prediction is already verified, we need to create a new one
        if latest_prediction and latest_prediction.get('verified'):
            print(f"[{now}] Latest prediction #{latest_prediction.get('id')} already verified, creating new prediction...")
            make_prediction()
            latest_prediction = dataC.get_latest_prediction()
            stats = dataC.get_prediction_stats()

        if latest_prediction and not latest_prediction.get('verified'):
            pr_id = latest_prediction.get('id')
            pr_lat = latest_prediction.get('predicted_lat')
            pr_lon = latest_prediction.get('predicted_lon')
            pr_mag = latest_prediction.get('predicted_mag')
            pr_dt = latest_prediction.get('predicted_dt') or 60
            pr_timestamp = datetime.fromisoformat(latest_prediction.get('timestamp'))

            # Calculate time window
            expected_event_time = pr_timestamp + timedelta(minutes=pr_dt)
            is_expired = now > expected_event_time

            # Calculate remaining/elapsed time
            if is_expired:
                elapsed_seconds = int((now - expected_event_time).total_seconds())
                time_remaining_seconds = -elapsed_seconds
            else:
                time_remaining_seconds = int((expected_event_time - now).total_seconds())

            # Build prediction status for frontend
            prediction_status = {
                'is_expired': is_expired,
                'time_remaining_seconds': time_remaining_seconds,
                'start_time': pr_timestamp.isoformat() + 'Z',
                'end_time': expected_event_time.isoformat() + 'Z',
                'current_time': now.isoformat() + 'Z'
            }

            # Values from get_latest_prediction() are already decoded
            pred_lat_actual = pr_lat  # already actual lat
            pred_lon_actual = pr_lon  # already actual lon
            pred_mag_actual = pr_mag or 4.0  # already actual mag

            # Calculate distance for each earthquake using haversine (km)
            if pred_lat_actual is not None and pred_lon_actual is not None:
                min_distance_km = float('inf')

                # Calculate distance and check match for each earthquake
                for eq in recent_earthquakes:
                    eq_lat = eq.get('lat')  # already decoded
                    eq_lon = eq.get('lon')  # already decoded
                    eq_mag = eq.get('mag') or 0

                    if eq_lat is not None and eq_lon is not None:
                        dist_km = haversine_km(pred_lat_actual, pred_lon_actual, eq_lat, eq_lon)
                        eq['distance_km'] = round(dist_km)
                        eq['distance'] = round(dist_km)

                        if dist_km < min_distance_km:
                            min_distance_km = dist_km

                    # Match = M4+ within 250km and after prediction time
                    eq_time_str = eq.get('time', '')
                    if eq_time_str and eq_mag >= MIN_MAG_DISPLAY and eq.get('distance_km') is not None:
                        eq_time = datetime.fromisoformat(eq_time_str.replace('Z', '+00:00')).replace(tzinfo=None)
                        if eq_time >= pr_timestamp and eq['distance_km'] <= MATCH_RADIUS_KM:
                            eq['is_match'] = True
                            if not matched_earthquake:
                                matched_earthquake = eq
                        else:
                            eq['is_match'] = False
                    else:
                        eq['is_match'] = False

                # Build closest_match info (closest M4+ earthquake regardless of match)
                closest_m4 = None
                for eq in recent_earthquakes:
                    if (eq.get('mag') or 0) >= MIN_MAG_DISPLAY and eq.get('distance_km') is not None:
                        if closest_m4 is None or eq['distance_km'] < closest_m4.get('distance_km', float('inf')):
                            closest_m4 = eq

                if matched_earthquake:
                    match_dist = matched_earthquake.get('distance_km', 0)
                    closest_match = {
                        'earthquake_id': matched_earthquake.get('id'),
                        'distance_km': match_dist,
                        'distance': match_dist,
                        'is_match': True,
                        'place': matched_earthquake.get('place'),
                        'mag': matched_earthquake.get('mag')
                    }
                    match_info = closest_match
                elif closest_m4:
                    closest_match = {
                        'earthquake_id': closest_m4.get('id'),
                        'distance_km': closest_m4.get('distance_km'),
                        'distance': closest_m4.get('distance_km'),
                        'is_match': False,
                        'place': closest_m4.get('place'),
                        'mag': closest_m4.get('mag')
                    }

            # AUTO-HANDLE: match found → verify and new prediction
            if matched_earthquake:
                eq_mag = matched_earthquake.get('mag')
                eq_dist = matched_earthquake.get('distance_km', 0)

                print(f"\n{'='*60}")
                print(f"[{now}] MATCHED via /api/live!")
                print(f"  Prediction #{pr_id}: {pred_lat_actual:.1f}°, {pred_lon_actual:.1f}°, M{pred_mag_actual:.1f}, dt={pr_dt}min")
                print(f"  Actual: M{eq_mag} - {matched_earthquake.get('place')} ({eq_dist}km away)")
                print(f"{'='*60}\n")

                dataC.update_prediction_match(
                    pr_id=pr_id,
                    earthquake_id=matched_earthquake.get('id'),
                    earthquake_lat=matched_earthquake.get('lat'),
                    earthquake_lon=matched_earthquake.get('lon'),
                    earthquake_mag=matched_earthquake.get('mag'),
                    earthquake_time=matched_earthquake.get('time'),
                    distance=eq_dist
                )

                print(f"[{now}] Creating new prediction...")
                make_prediction()

                latest_prediction = dataC.get_latest_prediction()
                recent_earthquakes = dataC.get_recent_earthquakes(limit=50, min_mag=2.0)
                stats = dataC.get_prediction_stats()

            # AUTO-HANDLE: window expired, no M4+ eq yet → MISSED
            elif is_expired:
                print(f"[{now}] Prediction #{pr_id} window expired (dt={pr_dt}min), marking MISSED...")

                dataC.verify_prediction(
                    pr_id=pr_id, actual_id=None,
                    actual_lat=None, actual_lon=None, actual_dt=None,
                    actual_mag=None, actual_time=None,
                    diff_lat=None, diff_lon=None, diff_dt=None,
                    diff_mag=None, correct=False
                )

                print(f"[{now}] Creating new prediction...")
                make_prediction()

                latest_prediction = dataC.get_latest_prediction()
                recent_earthquakes = dataC.get_recent_earthquakes(limit=50, min_mag=2.0)
                stats = dataC.get_prediction_stats()

                # Recalculate status for new prediction
                if latest_prediction and not latest_prediction.get('verified'):
                    new_pr_dt = latest_prediction.get('predicted_dt') or 60
                    new_pr_timestamp = datetime.fromisoformat(latest_prediction.get('timestamp'))
                    new_expected_time = new_pr_timestamp + timedelta(minutes=new_pr_dt)
                    prediction_status = {
                        'is_expired': False,
                        'time_remaining_seconds': int((new_expected_time - now).total_seconds()),
                        'start_time': new_pr_timestamp.isoformat() + 'Z',
                        'end_time': new_expected_time.isoformat() + 'Z',
                        'current_time': now.isoformat() + 'Z'
                    }

        # If prediction is verified, include the matched earthquake info
        if latest_prediction and latest_prediction.get('verified') and latest_prediction.get('correct'):
            match_info = {
                'verified_match': True,
                'actual_lat': latest_prediction.get('actual_lat'),
                'actual_lon': latest_prediction.get('actual_lon'),
                'actual_mag': latest_prediction.get('actual_mag'),
                'diff_lat': latest_prediction.get('diff_lat'),
                'diff_lon': latest_prediction.get('diff_lon')
            }

        return jsonify({
            'success': True,
            'server_time': now.isoformat(),
            'latest_prediction': latest_prediction,
            'recent_earthquakes': recent_earthquakes,
            'stats': stats,
            'match_info': match_info,
            'closest_match': closest_match,
            'prediction_status': prediction_status
        })

    except Exception as e:
        print(f"Error in /api/live: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get predictions with actual earthquakes (paginated)

    Query params:
        page: Page number (1-indexed, default 1)
        limit: Items per page (default 20, max 100)
        filter: Optional filter - 'matched', 'missed', 'pending', or empty for all
    """
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        page = request.args.get('page', 1, type=int)
        limit = min(request.args.get('limit', 20, type=int), 100)
        filter_type = request.args.get('filter', None)

        if filter_type == '':
            filter_type = None

        offset = (page - 1) * limit
        result = dataC.get_predictions_with_actuals(limit=limit, offset=offset, filter_type=filter_type)

        predictions = result['predictions']
        total = result['total']

        for p in predictions:
            if p.get('prediction_time') and hasattr(p['prediction_time'], 'isoformat'):
                p['prediction_time'] = p['prediction_time'].isoformat()
            if p.get('actual_time') and hasattr(p['actual_time'], 'isoformat'):
                p['actual_time'] = p['actual_time'].isoformat()

        total_pages = (total + limit - 1) // limit if limit > 0 else 1

        return jsonify({
            'success': True,
            'predictions': predictions,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Manually trigger a prediction"""
    pr_id = make_prediction()

    if pr_id:
        latest = dataC.get_latest_prediction()
        return jsonify({
            'success': True,
            'prediction': latest,
            'message': f'Prediction saved with ID {pr_id}'
        })
    else:
        return jsonify({'error': 'Failed to make prediction'}), 500


@app.route('/api/refresh', methods=['POST'])
def api_refresh_data():
    """Manually trigger USGS data refresh"""
    global dataC

    try:
        dataC.usgs2DB()
        return jsonify({
            'success': True,
            'message': 'USGS data refreshed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get prediction success statistics"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        stats = dataC.get_prediction_stats()
        result = dataC.get_predictions_with_actuals(limit=10)
        predictions = result.get('predictions', [])

        for p in predictions:
            if p.get('prediction_time') and hasattr(p['prediction_time'], 'isoformat'):
                p['prediction_time'] = p['prediction_time'].isoformat()
            if p.get('actual_time') and hasattr(p['actual_time'], 'isoformat'):
                p['actual_time'] = p['actual_time'].isoformat()

        return jsonify({
            'success': True,
            'stats': stats,
            'recent_predictions': predictions,
            'last_updated': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recent-earthquakes', methods=['GET'])
def recent_earthquakes():
    """Get recent earthquake data"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        limit = request.args.get('limit', 20, type=int)
        min_mag = request.args.get('min_mag', 4.0, type=float)
        earthquakes = dataC.get_recent_earthquakes(limit=limit, min_mag=min_mag)

        return jsonify({
            'success': True,
            'earthquakes': earthquakes,
            'count': len(earthquakes)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/earthquakes-24h', methods=['GET'])
def earthquakes_24h():
    """Get all earthquakes from the last 24 hours for the realtime map"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        min_mag = request.args.get('min_mag', 2.0, type=float)

        # Get earthquakes from last 24 hours
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
        WHERE us_datetime >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        AND us_mag >= %s
        AND us_type = 'earthquake'
        ORDER BY us_datetime DESC
        """

        rows = dataC._safe_fetch(sql, (min_mag,))

        earthquakes = []
        stats = {'total': 0, 'byMag': {'7+': 0, '6+': 0, '5+': 0, '4+': 0, '3+': 0, '<3': 0}}

        for row in rows:
            # Add 'Z' suffix to indicate UTC time for proper JS parsing
            time_str = row[1].isoformat() + 'Z' if row[1] else None
            eq = {
                'id': row[0],
                'time': time_str,
                'lat': float(row[2]) if row[2] is not None else None,
                'lon': float(row[3]) if row[3] is not None else None,
                'mag': float(row[4]) if row[4] is not None else None,
                'depth': float(row[5]) if row[5] is not None else None,
                'place': row[6]
            }
            earthquakes.append(eq)

            # Count by magnitude
            mag = eq['mag'] or 0
            if mag >= 7:
                stats['byMag']['7+'] += 1
            elif mag >= 6:
                stats['byMag']['6+'] += 1
            elif mag >= 5:
                stats['byMag']['5+'] += 1
            elif mag >= 4:
                stats['byMag']['4+'] += 1
            elif mag >= 3:
                stats['byMag']['3+'] += 1
            else:
                stats['byMag']['<3'] += 1

        stats['total'] = len(earthquakes)

        return jsonify({
            'success': True,
            'earthquakes': earthquakes,
            'stats': stats,
            'count': len(earthquakes),
            'period': '24h',
            'min_mag': min_mag,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in /api/earthquakes-24h: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/<int:prediction_id>', methods=['GET'])
def get_prediction_detail(prediction_id):
    """Get detailed prediction info including earthquakes in its time window"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        # Get the prediction
        sql = """
        SELECT
            p.pr_id, p.pr_timestamp,
            p.pr_lat_predicted, p.pr_lon_predicted, p.pr_dt_predicted, p.pr_mag_predicted, p.pr_place,
            p.pr_lat_actual, p.pr_lon_actual, p.pr_dt_actual, p.pr_mag_actual,
            p.pr_diff_lat, p.pr_diff_lon, p.pr_diff_dt, p.pr_diff_mag,
            p.pr_verified, p.pr_correct, p.pr_actual_time,
            u.us_place as actual_place
        FROM predictions p
        LEFT JOIN usgs u ON p.pr_actual_id = u.us_id
        WHERE p.pr_id = %s
        """
        row = dataC._safe_fetch(sql, (prediction_id,), fetch_one=True)

        if not row:
            return jsonify({'error': 'Prediction not found'}), 404

        pr_timestamp = row[1]
        pr_dt = row[4] or 60  # Default 60 minutes if not set

        prediction = {
            'id': row[0],
            'prediction_time': pr_timestamp.isoformat() if pr_timestamp else None,
            'predicted_lat': (row[2] - 90) if row[2] else None,  # Decode to actual lat
            'predicted_lon': (row[3] - 180) if row[3] else None,  # Decode to actual lon
            'predicted_dt': pr_dt,
            'predicted_mag': row[5] / 10.0 if row[5] else None,  # Decode magnitude
            'predicted_place': row[6],
            'actual_lat': (row[7] - 90) if row[7] else None,
            'actual_lon': (row[8] - 180) if row[8] else None,
            'actual_dt': row[9],
            'actual_mag': row[10] / 10.0 if row[10] else None,
            'actual_place': row[18],  # From JOIN
            'diff_lat': row[11],
            'diff_lon': row[12],
            'diff_dt': row[13],
            'diff_mag': row[14],
            'verified': bool(row[15]),
            'correct': bool(row[16]),
            'actual_time': row[17].isoformat() if row[17] else None,
        }

        # Calculate prediction window
        window_end = pr_timestamp + timedelta(minutes=pr_dt) if pr_timestamp else None

        prediction['window_start'] = pr_timestamp.isoformat() if pr_timestamp else None
        prediction['window_end'] = window_end.isoformat() if window_end else None

        # Get all earthquakes that occurred during the prediction window
        # Extend window to 24 hours to show context
        extended_end = pr_timestamp + timedelta(hours=24) if pr_timestamp else None

        earthquakes_sql = """
        SELECT us_id, us_datetime, us_x, us_y, us_mag, us_place, us_m
        FROM usgs
        WHERE us_datetime BETWEEN %s AND %s
        AND us_mag >= 2.0
        AND us_type = 'earthquake'
        ORDER BY us_datetime ASC
        """
        eq_rows = dataC._safe_fetch(earthquakes_sql, (pr_timestamp, extended_end))

        earthquakes = []
        for eq in eq_rows:
            eq_time = eq[1]
            minutes_after_prediction = int((eq_time - pr_timestamp).total_seconds() / 60) if eq_time and pr_timestamp else None

            # Check if this earthquake is within the prediction window
            in_window = minutes_after_prediction is not None and 0 <= minutes_after_prediction <= pr_dt

            # Match = M4+ earthquake within 250km and within prediction time window
            pr_lat_encoded = row[2]  # Still encoded
            pr_lon_encoded = row[3]
            eq_lat_encoded = eq[2]
            eq_lon_encoded = eq[3]

            is_match = False
            distance = None
            if pr_lat_encoded and pr_lon_encoded and eq_lat_encoded and eq_lon_encoded:
                dist_km = haversine_km(pr_lat_encoded - 90, pr_lon_encoded - 180,
                                       eq_lat_encoded - 90, eq_lon_encoded - 180)
                distance = round(dist_km)
                is_match = in_window and (eq[4] or 0) >= MIN_MAG_DISPLAY and dist_km <= MATCH_RADIUS_KM

            earthquakes.append({
                'id': eq[0],
                'time': eq_time.isoformat() if eq_time else None,
                'lat': eq_lat_encoded - 90 if eq_lat_encoded else None,  # Decode
                'lon': eq_lon_encoded - 180 if eq_lon_encoded else None,  # Decode
                'mag': eq[4],
                'place': eq[5],
                'depth': eq[6],
                'minutes_after': minutes_after_prediction,
                'in_window': in_window,
                'is_match': is_match,
                'distance': round(distance, 1) if distance else None,
            })

        return jsonify({
            'success': True,
            'prediction': prediction,
            'earthquakes': earthquakes,
            'earthquake_count': len(earthquakes),
            'window_earthquakes': len([e for e in earthquakes if e['in_window']]),
            'matching_earthquakes': len([e for e in earthquakes if e['is_match']]),
        })

    except Exception as e:
        print(f"Error getting prediction detail: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def get_training_info():
    """Get training info from training_status.json or train.out"""
    training_info = {
        'latest_step': None,
        'latest_loss': None,
        'checkpoint_step': None,
        'checkpoint_loss': None,
    }

    # First try reading from training_status.json (preferred, written by train.py)
    status_file = os.path.join(MODEL_DIR, 'training_status.json')
    try:
        if os.path.exists(status_file):
            import json
            with open(status_file, 'r') as f:
                status = json.load(f)
                training_info['latest_step'] = status.get('latest_step')
                training_info['latest_loss'] = status.get('latest_loss')
                training_info['checkpoint_step'] = status.get('checkpoint_step')
                training_info['checkpoint_loss'] = status.get('checkpoint_loss')
                # If we got data, return it
                if training_info['latest_step'] is not None:
                    return training_info
    except Exception as e:
        print(f"Error reading training_status.json: {e}")

    # Fallback: parse train.out
    train_out_path = os.path.join(MODEL_DIR, 'train.out')
    try:
        if os.path.exists(train_out_path) and os.path.getsize(train_out_path) > 0:
            import subprocess
            # Get last 100 lines efficiently
            result = subprocess.run(
                ['tail', '-100', train_out_path],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')

            # Parse from bottom up to find latest info
            for line in reversed(lines):
                # Look for step line: "step 693200 | loss 16.3251"
                if line.startswith('step ') and training_info['latest_step'] is None:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        step_part = parts[0].strip()  # "step 693200"
                        loss_part = parts[1].strip()  # "loss 16.3251"
                        try:
                            training_info['latest_step'] = int(step_part.replace('step ', ''))
                            training_info['latest_loss'] = float(loss_part.replace('loss ', ''))
                        except:
                            pass

                # Look for checkpoint line: "Checkpoint saved: ... (iter=693000, loss=16.3798)"
                if 'Checkpoint saved:' in line and training_info['checkpoint_step'] is None:
                    import re
                    match = re.search(r'iter=(\d+),\s*loss=([\d.]+)', line)
                    if match:
                        training_info['checkpoint_step'] = int(match.group(1))
                        training_info['checkpoint_loss'] = float(match.group(2))

                # Stop if we have all info
                if training_info['latest_step'] and training_info['checkpoint_step']:
                    break

    except Exception as e:
        print(f"Error parsing train.out: {e}")

    return training_info


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get model and server status"""
    global current_checkpoint

    checkpoint_time = None
    checkpoint_exists = current_checkpoint and os.path.exists(current_checkpoint)
    if checkpoint_exists:
        checkpoint_time = datetime.fromtimestamp(os.path.getmtime(current_checkpoint)).isoformat()

    # Get training info from train.out
    training_info = get_training_info()

    # Convert to camelCase for frontend
    training_camel = {
        'latestStep': training_info.get('latest_step'),
        'latestLoss': training_info.get('latest_loss'),
        'checkpointStep': training_info.get('checkpoint_step'),
        'checkpointLoss': training_info.get('checkpoint_loss'),
    }

    return jsonify({
        'loaded': model is not None,
        'device': device,
        'model_type': 'EqModelComplex',
        'current_checkpoint': os.path.basename(current_checkpoint) if checkpoint_exists else None,
        'checkpoint_time': checkpoint_time,
        'training': training_camel,
        'config': {
            'sequence_length': T,
            'embedding_size': n_embed,
            'num_heads': n_heads,
            'num_layers': n_layer,
            'match_radius_km': MATCH_RADIUS_KM,
            'min_mag_display': MIN_MAG_DISPLAY
        },
        'scheduler_running': scheduler is not None and scheduler.running if scheduler else False
    })


@app.route('/api/reload-model', methods=['POST'])
def reload_model():
    """Manually reload model from latest checkpoint"""
    try:
        load_model(force_reload=True)
        return jsonify({
            'success': True,
            'checkpoint': os.path.basename(current_checkpoint) if current_checkpoint else None,
            'message': 'Model reloaded'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training-loss', methods=['GET'])
def get_training_loss():
    """Get training loss history for web UI graph"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        # Get limit parameter (default 500 points for graph)
        limit = request.args.get('limit', 500, type=int)

        # Query loss history from database
        sql = """
        SELECT step, train_loss, val_loss, timestamp
        FROM training_loss
        ORDER BY step DESC
        LIMIT %s
        """
        rows = dataC._safe_fetch(sql, (limit,))

        if not rows:
            return jsonify({'loss_history': [], 'total': 0})

        # Convert to list of dicts (reverse to get ascending order)
        loss_history = []
        for row in reversed(rows):
            loss_history.append({
                'step': row[0],
                'train_loss': float(row[1]) if row[1] else None,
                'val_loss': float(row[2]) if row[2] else None,
                'timestamp': row[3].isoformat() if row[3] else None
            })

        return jsonify({
            'loss_history': loss_history,
            'total': len(loss_history)
        })

    except Exception as e:
        print(f"Error getting training loss: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cycle', methods=['POST'])
def trigger_cycle():
    """Manually trigger a monitoring cycle"""
    try:
        monitor_cycle()
        return jsonify({
            'success': True,
            'message': 'Monitor cycle completed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/match', methods=['POST'])
def record_match():
    """Record a real-time match detected by the frontend"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        earthquake_id = data.get('earthquake_id')
        earthquake_lat = data.get('earthquake_lat')
        earthquake_lon = data.get('earthquake_lon')
        earthquake_mag = data.get('earthquake_mag')
        earthquake_time = data.get('earthquake_time')
        distance = data.get('distance')

        if not all([prediction_id, earthquake_id]):
            return jsonify({'error': 'Missing required fields'}), 400

        success = dataC.update_prediction_match(
            pr_id=prediction_id,
            earthquake_id=earthquake_id,
            earthquake_lat=earthquake_lat,
            earthquake_lon=earthquake_lon,
            earthquake_mag=earthquake_mag,
            earthquake_time=earthquake_time,
            distance=distance
        )

        if success:
            print(f"[{datetime.now()}] Match recorded via API for prediction #{prediction_id}")

            # Trigger new prediction after recording match
            print(f"[{datetime.now()}] Starting new prediction after match...")
            new_pred = make_prediction()

            # Get updated stats
            stats = dataC.get_prediction_stats()
            return jsonify({
                'success': True,
                'message': f'Match recorded for prediction {prediction_id}',
                'stats': stats,
                'new_prediction': new_pred
            })
        else:
            return jsonify({'error': 'Failed to update prediction'}), 500

    except Exception as e:
        print(f"Error recording match: {e}")
        return jsonify({'error': str(e)}), 500


# ============= INITIALIZATION =============

_initialized = False

def init_app():
    """Initialize the application - non-blocking"""
    global _initialized, dataC

    if _initialized:
        return

    _initialized = True

    print("=" * 60)
    print("EARTHQUAKE PREDICTION SERVER")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: n_embed={n_embed}, T={T}, heads={n_heads}, layers={n_layer}")
    print(f"Hybrid: M{INPUT_MAG}+ input, M{TARGET_MAG}+ targets")

    # Load data and model
    print("\nLoading hybrid data...")
    dataC = DataC()
    dataC.getDataHybrid(input_mag=INPUT_MAG, target_mag=TARGET_MAG)

    print("\nLoading model...")
    load_model()

    # Check if we have an active prediction, if not create one
    print("\nChecking for active prediction...")
    latest = dataC.get_latest_prediction(min_mag=4.0)
    if not latest or latest.get('verified'):
        print("No active prediction found, creating initial prediction...")
        make_prediction()
    else:
        # Check if the existing prediction has expired
        pr_timestamp = datetime.fromisoformat(latest['timestamp'])
        pr_dt = latest.get('predicted_dt') or 60
        expected_event_time = pr_timestamp + timedelta(minutes=pr_dt)

        if datetime.now() > expected_event_time:
            print(f"Active prediction #{latest['id']} has expired, creating new prediction...")
            make_prediction()
        else:
            remaining = (expected_event_time - datetime.now()).total_seconds() / 60
            print(f"Active prediction found: #{latest['id']} ({remaining:.1f} minutes remaining)")

    print("\nStarting scheduler...")
    start_scheduler()

    print("=" * 60)
    print("Server ready! Monitoring every 120 seconds.")
    print("=" * 60)


# Initialize on module load
init_app()


if __name__ == '__main__':
    print(f"Server starting on http://0.0.0.0:3000")
    try:
        app.run(host='0.0.0.0', port=3000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if scheduler:
            scheduler.shutdown()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] Server crashed: {e}")
        import traceback
        traceback.print_exc()
        if scheduler:
            scheduler.shutdown()
        sys.exit(1)
