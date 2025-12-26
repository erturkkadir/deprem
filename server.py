"""
Earthquake Prediction Server
- Loads latest checkpoint from training
- Makes predictions every 5 minutes
- Saves predictions to database
- Auto-verifies predictions against actual earthquakes
- Serves API for frontend on port 3000
"""
import os
import glob
import math
import torch
import atexit
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from torch.nn import functional as F

from DataClass import DataC
from EqModel import ComplexEqModel


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
n_layer = 8
dropout = 0.01
INPUT_MAG = 2.0   # Min magnitude for input context
TARGET_MAG = 4.0  # Min magnitude for targets

# Global state
model = None
dataC = None
scheduler = None
current_checkpoint = None  # Track which checkpoint is loaded

MODEL_DIR = '/var/www/syshuman/quake'
MODEL_PREFIX = 'eqModel_complex'

# Verification tolerances
DISTANCE_RADIUS = 15  # degrees - circular distance for matching
DT_TOLERANCE = 30     # minutes
MAG_TOLERANCE = 1.0   # magnitude
MIN_MAG_DISPLAY = 4.0 # Only show predictions with mag >= 4.0


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

    model = ComplexEqModel(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max)
    model.to(device)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Clean keys from torch.compile prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")

        current_checkpoint = checkpoint_path
        print(f"[{datetime.now()}] Loaded checkpoint: {os.path.basename(checkpoint_path)}")
    else:
        print("No checkpoint found, using random weights")

    model.eval()
    return True


def reload_if_new_checkpoint():
    """Check for new checkpoint and reload if found"""
    global current_checkpoint

    latest = get_latest_checkpoint()
    if latest and latest != current_checkpoint:
        print(f"[{datetime.now()}] New checkpoint detected: {os.path.basename(latest)}")
        load_model(force_reload=True)
        return True
    return False


def make_prediction():
    """Make a prediction and save to database"""
    global model, dataC

    if model is None or dataC is None:
        print("Model or data not loaded")
        return None

    try:
        # Reload data to get latest earthquakes (hybrid: M2+ input, M4+ targets)
        dataC.getDataHybrid(input_mag=INPUT_MAG, target_mag=TARGET_MAG)

        # Get last sequence from data
        x_test, y_actual = dataC.getLast(1, T, 'val', col=col)
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
    """Verify old predictions against actual earthquakes"""
    global dataC

    if dataC is None:
        return

    try:
        unverified = dataC.get_unverified_predictions(older_than_hours=24)

        for pred in unverified:
            pr_id, pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted = pred

            start_time = pr_timestamp
            end_time = pr_timestamp + timedelta(hours=24)

            actuals = dataC.get_earthquakes_in_window(start_time, end_time, min_mag=4.0)

            if actuals:
                # Find closest match by circular distance
                best_match = None
                min_distance = float('inf')

                for actual in actuals:
                    us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place = actual
                    lat_diff = abs(pr_lat_predicted - us_x) if pr_lat_predicted else 180
                    lon_diff = abs(pr_lon_predicted - us_y) if pr_lon_predicted else 360
                    lon_diff = min(lon_diff, 360 - lon_diff)

                    distance = math.sqrt(lat_diff ** 2 + lon_diff ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = actual

                if best_match:
                    us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place = best_match

                    diff_lat = abs(pr_lat_predicted - us_x) if pr_lat_predicted else None
                    diff_lon = abs(pr_lon_predicted - us_y) if pr_lon_predicted else None
                    if diff_lon:
                        diff_lon = min(diff_lon, 360 - diff_lon)

                    actual_dt = int((us_datetime - pr_timestamp).total_seconds() / 60) if us_datetime else None
                    diff_dt = abs(pr_dt_predicted - actual_dt) if (pr_dt_predicted and actual_dt) else None

                    actual_mag_encoded = int(us_mag * 10) if us_mag else None
                    diff_mag = abs((pr_mag_predicted / 10.0) - us_mag) if (pr_mag_predicted and us_mag) else None

                    circular_distance = math.sqrt((diff_lat or 0) ** 2 + (diff_lon or 0) ** 2)
                    correct = circular_distance <= DISTANCE_RADIUS

                    dataC.verify_prediction(
                        pr_id=pr_id,
                        actual_id=us_id,
                        actual_lat=us_x,
                        actual_lon=us_y,
                        actual_dt=actual_dt,
                        actual_mag=actual_mag_encoded,
                        actual_time=us_datetime,
                        diff_lat=int(diff_lat) if diff_lat else None,
                        diff_lon=int(diff_lon) if diff_lon else None,
                        diff_dt=int(diff_dt) if diff_dt else None,
                        diff_mag=diff_mag,
                        correct=correct
                    )
                    print(f"[{datetime.now()}] Verified prediction {pr_id}: distance={circular_distance:.1f}, correct={correct}")

    except Exception as e:
        print(f"Error in auto-verification: {e}")
        import traceback
        traceback.print_exc()


def check_and_handle_prediction():
    """
    Main prediction cycle logic:
    1. Check if current prediction has a match -> verify & new prediction
    2. Check if current prediction expired -> new prediction
    3. If no active prediction -> new prediction

    Returns: dict with status info or None
    """
    global dataC

    if dataC is None:
        return None

    try:
        dataC._ensure_connection()

        # Get the latest prediction
        latest = dataC.get_latest_prediction(min_mag=4.0)

        # No prediction exists - make one
        if not latest:
            print(f"[{datetime.now()}] No active prediction, creating new one...")
            make_prediction()
            return {'action': 'new_prediction', 'reason': 'no_prediction'}

        # Prediction already verified - make new one
        if latest.get('verified'):
            print(f"[{datetime.now()}] Previous prediction verified, creating new one...")
            make_prediction()
            return {'action': 'new_prediction', 'reason': 'previous_verified'}

        # Active unverified prediction - check for match or expiry
        pr_id = latest['id']
        pr_timestamp = datetime.fromisoformat(latest['timestamp'])
        pr_lat = latest.get('predicted_lat')
        pr_lon = latest.get('predicted_lon')
        pr_mag = latest.get('predicted_mag')
        pr_dt = latest.get('predicted_dt') or 60  # Default 60 min if not set

        if pr_lat is None or pr_lon is None:
            return None

        # Get recent earthquakes since prediction was made
        end_time = datetime.now()
        recent_quakes = dataC.get_earthquakes_in_window(pr_timestamp, end_time, min_mag=4.0)

        # Check for match
        if recent_quakes:
            for quake in recent_quakes:
                us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place = quake

                # Calculate circular distance
                # pr_lat/lon are actual coords (-90 to 90, -180 to 180)
                # us_x/y are encoded (0-180, 0-360), so convert pr to encoded
                lat_diff = abs((pr_lat + 90) - us_x)
                lon_diff = abs((pr_lon + 180) - us_y)
                lon_diff = min(lon_diff, 360 - lon_diff)
                distance = math.sqrt(lat_diff ** 2 + lon_diff ** 2)

                if distance <= DISTANCE_RADIUS:
                    # MATCH FOUND!
                    print(f"\n{'='*60}")
                    print(f"[{datetime.now()}] MATCH FOUND!")
                    print(f"  Prediction #{pr_id}: {pr_lat:.1f}°, {pr_lon:.1f}°")
                    print(f"  Actual: {us_x-90:.1f}°, {us_y-180:.1f}° - M{us_mag} - {us_place}")
                    print(f"  Distance: {distance:.1f}° (threshold: {DISTANCE_RADIUS}°)")
                    print(f"{'='*60}\n")

                    # Calculate differences
                    actual_dt = int((us_datetime - pr_timestamp).total_seconds() / 60)
                    diff_dt = abs(pr_dt - actual_dt) if pr_dt else None
                    diff_mag = abs(pr_mag - us_mag) if pr_mag and us_mag else None

                    # Update prediction as verified and correct
                    dataC.verify_prediction(
                        pr_id=pr_id,
                        actual_id=us_id,
                        actual_lat=us_x,
                        actual_lon=us_y,
                        actual_dt=actual_dt,
                        actual_mag=int(us_mag * 10) if us_mag else None,
                        actual_time=us_datetime,
                        diff_lat=int(lat_diff),
                        diff_lon=int(lon_diff),
                        diff_dt=int(diff_dt) if diff_dt else None,
                        diff_mag=diff_mag,
                        correct=True
                    )

                    # Make new prediction
                    print(f"[{datetime.now()}] Starting new prediction after match...")
                    make_prediction()

                    return {
                        'action': 'match_found',
                        'prediction_id': pr_id,
                        'earthquake_id': us_id,
                        'distance': distance,
                        'place': us_place,
                        'mag': us_mag
                    }

        # Check if prediction window has expired
        expected_event_time = pr_timestamp + timedelta(minutes=pr_dt)
        if datetime.now() > expected_event_time:
            print(f"[{datetime.now()}] Prediction #{pr_id} expired (waited {pr_dt} min), creating new prediction...")
            # Make new prediction (old one will be verified later by auto_verify)
            make_prediction()
            return {'action': 'expired', 'prediction_id': pr_id}

        # Still waiting - no action needed
        remaining = (expected_event_time - datetime.now()).total_seconds() / 60
        return {'action': 'waiting', 'prediction_id': pr_id, 'remaining_minutes': round(remaining, 1)}

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
        try:
            dataC = DataC()
            dataC.getDataHybrid(input_mag=INPUT_MAG, target_mag=TARGET_MAG)
        except:
            pass
        return False


def monitor_cycle():
    """
    Main monitoring loop - runs every 60 seconds:
    1. Quick refresh USGS data (last day only)
    2. Check prediction status (match/expire/waiting)
    """
    global dataC

    # Run within Flask app context
    with app.app_context():
        try:
            # Check for new model checkpoint
            reload_if_new_checkpoint()

            # Quick USGS data refresh (just update DB)
            if dataC is None:
                dataC = DataC()
            dataC._ensure_connection()
            dataC.usgs2DB(days=1)

            # Check and handle current prediction
            result = check_and_handle_prediction()
            if result:
                action = result.get('action')
                print(f"[{datetime.now()}] Status: {action} - {result}")

            # Verify old predictions (>24 hours)
            auto_verify_predictions()

        except Exception as e:
            print(f"Error in monitor cycle: {e}")
            try:
                dataC = DataC()
                dataC.getDataHybrid(input_mag=INPUT_MAG, target_mag=TARGET_MAG)
            except:
                pass


def start_scheduler():
    """Start background scheduler"""
    global scheduler

    scheduler = BackgroundScheduler()

    # Single monitoring job every 60 seconds (gives enough time to complete)
    scheduler.add_job(func=monitor_cycle, trigger="interval", seconds=60, id='monitor_cycle')

    scheduler.start()
    print("Scheduler started: monitoring every 60 seconds")

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
    return send_from_directory(app.static_folder, path)


@app.route('/api/live', methods=['GET'])
def get_live_data():
    """Get live data: latest prediction, recent earthquakes, stats, and match info"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        latest_prediction = dataC.get_latest_prediction()
        recent_earthquakes = dataC.get_recent_earthquakes(limit=50, min_mag=2.0)
        stats = dataC.get_prediction_stats()

        # Calculate match info for each earthquake (so frontend doesn't need to)
        match_info = None
        closest_match = None

        if latest_prediction and not latest_prediction.get('verified'):
            pr_lat = latest_prediction.get('predicted_lat')
            pr_lon = latest_prediction.get('predicted_lon')

            if pr_lat is not None and pr_lon is not None:
                min_distance = float('inf')

                for eq in recent_earthquakes:
                    eq_lat = eq.get('lat')
                    eq_lon = eq.get('lon')
                    eq_mag = eq.get('mag') or 0

                    if eq_lat is not None and eq_lon is not None:
                        lat_diff = abs(pr_lat - eq_lat)
                        lon_diff = abs(pr_lon - eq_lon)
                        lon_diff = min(lon_diff, 360 - lon_diff)
                        distance = math.sqrt(lat_diff ** 2 + lon_diff ** 2)

                        eq['distance'] = round(distance, 1)
                        # Only M4+ earthquakes can be matches
                        eq['is_match'] = distance <= DISTANCE_RADIUS and eq_mag >= MIN_MAG_DISPLAY

                        # Track closest M4+ earthquake
                        if eq_mag >= MIN_MAG_DISPLAY and distance < min_distance:
                            min_distance = distance
                            closest_match = {
                                'earthquake_id': eq.get('id'),
                                'distance': round(distance, 1),
                                'is_match': distance <= DISTANCE_RADIUS,
                                'place': eq.get('place'),
                                'mag': eq_mag
                            }

                if closest_match and closest_match['is_match']:
                    match_info = closest_match

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
            'timestamp': datetime.now().isoformat(),
            'latest_prediction': latest_prediction,
            'recent_earthquakes': recent_earthquakes,
            'stats': stats,
            'match_info': match_info,
            'closest_match': closest_match
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get predictions with actual earthquakes"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        limit = request.args.get('limit', 50, type=int)
        predictions = dataC.get_predictions_with_actuals(limit=limit)

        for p in predictions:
            if p.get('prediction_time') and hasattr(p['prediction_time'], 'isoformat'):
                p['prediction_time'] = p['prediction_time'].isoformat()
            if p.get('actual_time') and hasattr(p['actual_time'], 'isoformat'):
                p['actual_time'] = p['actual_time'].isoformat()

        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
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
        dataC._ensure_connection()
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
        predictions = dataC.get_predictions_with_actuals(limit=10)

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


@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get model and server status"""
    global current_checkpoint

    checkpoint_time = None
    if current_checkpoint:
        checkpoint_time = datetime.fromtimestamp(os.path.getmtime(current_checkpoint)).isoformat()

    return jsonify({
        'loaded': model is not None,
        'device': device,
        'model_type': 'ComplexEqModel',
        'current_checkpoint': os.path.basename(current_checkpoint) if current_checkpoint else None,
        'checkpoint_time': checkpoint_time,
        'config': {
            'sequence_length': T,
            'embedding_size': n_embed,
            'num_heads': n_heads,
            'num_layers': n_layer,
            'distance_radius': DISTANCE_RADIUS,
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
            # Get updated stats
            stats = dataC.get_prediction_stats()
            return jsonify({
                'success': True,
                'message': f'Match recorded for prediction {prediction_id}',
                'stats': stats
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
    print("Server ready! Monitoring every 30 seconds.")
    print("=" * 60)


# Initialize on module load
init_app()


if __name__ == '__main__':
    print(f"Server starting on http://0.0.0.0:3000")
    app.run(host='0.0.0.0', port=3000, debug=False, threaded=False, use_reloader=False)
