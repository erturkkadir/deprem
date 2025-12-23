"""
Flask API Server for Earthquake Prediction System
- Automated USGS data pull every 5 minutes
- Auto-verification of predictions against actual earthquakes
- Continuous training with automatic checkpoint management
- Serves API on port 1977 for external access
"""
import os
import glob
import json
import torch
import atexit
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from torch.nn import functional as F

from DataClass import DataC
from EqModel import ComplexEqModel, EqModel


def reverse_geocode(lat, lon):
    """Convert lat/lon to location name using OpenStreetMap Nominatim"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'zoom': 5,  # Country/state level
            'addressdetails': 1
        }
        headers = {'User-Agent': 'EarthquakePredictionSystem/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()

            # Build location string from address components
            address = data.get('address', {})
            parts = []

            # Try to get meaningful location parts
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

            # Fallback to display_name
            display = data.get('display_name', '')
            if display:
                # Take first few parts of the display name
                parts = display.split(', ')[:3]
                return ', '.join(parts)

        return None
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
        return None

app = Flask(__name__, static_folder='web/dist')
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Model configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
col = 2  # Latitude prediction
p_max = 180
B = 1
T = 1024
C = 1190
n_embed = C
n_heads = 8
n_layer = 8
dropout = 0.01

# Global state
model = None
dataC = None
scheduler = None

MODEL_DIR = '/var/www/syshuman/quake'
MODEL_PREFIX = 'eqModel_complex'
MODEL_PATH = f'{MODEL_DIR}/{MODEL_PREFIX}.pth'  # Legacy path
LATITUDE_TOLERANCE = 10  # degrees
MIN_MAG_DISPLAY = 4.0  # Only show predictions with mag >= 4.0

# Training configuration
TRAINING_ITERS = 200  # Training iterations per cycle
TRAINING_LR = 3e-4
TRAINING_INTERVAL_HOURS = 1  # Run training every hour


def get_latest_checkpoint():
    """Find the latest checkpoint file by timestamp"""
    # Look for timestamped checkpoints first
    pattern = f'{MODEL_DIR}/{MODEL_PREFIX}_*.pth'
    checkpoints = glob.glob(pattern)

    if checkpoints:
        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]

    # Fall back to legacy path
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    return None


def save_checkpoint(model_to_save):
    """Save model checkpoint with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = f'{MODEL_DIR}/{MODEL_PREFIX}_{timestamp}.pth'

    # Get the raw model if it's compiled
    if hasattr(model_to_save, '_orig_mod'):
        state_dict = model_to_save._orig_mod.state_dict()
    else:
        state_dict = model_to_save.state_dict()

    # Remove '_orig_mod.' prefix if present
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        clean_state_dict[new_key] = value

    torch.save(clean_state_dict, checkpoint_path)
    print(f"[{datetime.now()}] Checkpoint saved: {checkpoint_path}")

    # Also save to legacy path for compatibility
    torch.save(clean_state_dict, MODEL_PATH)

    # Keep only last 5 checkpoints
    cleanup_old_checkpoints(keep=5)

    return checkpoint_path


def cleanup_old_checkpoints(keep=5):
    """Remove old checkpoints, keeping only the most recent ones"""
    pattern = f'{MODEL_DIR}/{MODEL_PREFIX}_*.pth'
    checkpoints = glob.glob(pattern)

    if len(checkpoints) > keep:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        for old_checkpoint in checkpoints[keep:]:
            try:
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                print(f"Error removing checkpoint {old_checkpoint}: {e}")


def load_model():
    """Load or initialize the earthquake prediction model"""
    global model, dataC

    dataC = DataC()
    dataC.getData()
    sizes = dataC.getSizes()

    # Use ComplexEqModel as specified in CLAUDE.md
    model = ComplexEqModel(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max)
    model.to(device)

    checkpoint_path = get_latest_checkpoint()

    if checkpoint_path:
        # Load state dict and handle torch.compile() prefix
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Remove '_orig_mod.' prefix if present (from torch.compile)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value

        # Load with strict=False to allow missing keys (new prediction heads)
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys (will use random init): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("No saved model found, using initialized weights")

    model.eval()


def make_prediction():
    """Make a prediction and save to database"""
    global model, dataC

    if model is None or dataC is None:
        print("Model or data not loaded")
        return None

    try:
        # Reload data to get latest earthquakes
        dataC.getData()

        # Get last sequence from data
        x_test, y_actual = dataC.getLast(1, T, 'val', col=col)
        x_test = x_test.to(device)

        with torch.no_grad():
            # Use generate_multi for all 4 predictions
            predictions = model.generate_multi(x_test)

        lat_encoded = predictions['lat']   # 0-180 encoded
        lon_encoded = predictions['lon']   # 0-360 encoded
        dt_minutes = predictions['dt']     # 0-150 minutes
        mag_encoded = predictions['mag']   # 0-91 (mag * 10)

        # Convert to actual coordinates
        lat_actual = lat_encoded - 90
        lon_actual = lon_encoded - 180
        mag_actual = mag_encoded / 10.0

        # Get location name from coordinates
        place = reverse_geocode(lat_actual, lon_actual)

        # Save prediction to database with place
        pr_id = dataC.save_prediction(lat_encoded, lon_encoded, dt_minutes, mag_encoded, place)

        place_str = f" near {place}" if place else ""
        print(f"[{datetime.now()}] Prediction made: lat={lat_actual}°, lon={lon_actual}°{place_str}, dt={dt_minutes}min, mag={mag_actual} (id={pr_id})")
        return pr_id

    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


LONGITUDE_TOLERANCE = 20  # degrees for longitude (not used with circular matching)
DT_TOLERANCE = 30  # minutes for time difference
MAG_TOLERANCE = 1.0  # magnitude difference
DISTANCE_RADIUS = 15  # degrees - circular distance radius for matching sqrt(lat² + lon²)

def auto_verify_predictions():
    """Auto-verify predictions against actual earthquakes using circular distance matching"""
    global dataC
    import math

    if dataC is None:
        return

    try:
        # Get unverified predictions older than 24 hours
        unverified = dataC.get_unverified_predictions(older_than_hours=24)

        for pred in unverified:
            pr_id, pr_timestamp, pr_lat_predicted, pr_lon_predicted, pr_dt_predicted, pr_mag_predicted = pred

            # Look for actual earthquakes in the 24 hours after prediction
            start_time = pr_timestamp
            end_time = pr_timestamp + timedelta(hours=24)

            actuals = dataC.get_earthquakes_in_window(start_time, end_time, min_mag=4.0)

            if actuals:
                # Find the earthquake with closest circular distance
                best_match = None
                min_distance = float('inf')

                for actual in actuals:
                    us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place = actual
                    # Calculate differences
                    lat_diff = abs(pr_lat_predicted - us_x) if pr_lat_predicted else 180
                    lon_diff = abs(pr_lon_predicted - us_y) if pr_lon_predicted else 360
                    # Handle longitude wrap-around
                    lon_diff = min(lon_diff, 360 - lon_diff)

                    # Circular distance: sqrt(lat² + lon²)
                    distance = math.sqrt(lat_diff ** 2 + lon_diff ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = actual

                if best_match:
                    us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place = best_match

                    # Calculate all differences
                    diff_lat = abs(pr_lat_predicted - us_x) if pr_lat_predicted else None
                    diff_lon = abs(pr_lon_predicted - us_y) if pr_lon_predicted else None
                    if diff_lon:
                        diff_lon = min(diff_lon, 360 - diff_lon)

                    # Time difference: compare predicted dt with actual time since prediction
                    actual_dt = int((us_datetime - pr_timestamp).total_seconds() / 60) if us_datetime else None
                    diff_dt = abs(pr_dt_predicted - actual_dt) if (pr_dt_predicted and actual_dt) else None

                    # Magnitude difference
                    actual_mag_encoded = int(us_mag * 10) if us_mag else None
                    diff_mag = abs((pr_mag_predicted / 10.0) - us_mag) if (pr_mag_predicted and us_mag) else None

                    # Circular distance matching: sqrt(lat² + lon²) <= DISTANCE_RADIUS
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
                    print(f"[{datetime.now()}] Verified prediction {pr_id}: distance={circular_distance:.1f}° (radius={DISTANCE_RADIUS}°), matched={correct}")

    except Exception as e:
        print(f"Error in auto-verification: {e}")
        import traceback
        traceback.print_exc()


def run_training_cycle():
    """Run a training cycle to continuously improve the model"""
    global model, dataC

    print(f"\n[{datetime.now()}] Starting training cycle...")

    try:
        # Reload latest data
        dataC = DataC()
        dataC.getData()

        # Switch model to training mode
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_LR)
        total_loss = 0
        num_batches = 0

        for iter in range(TRAINING_ITERS):
            optimizer.zero_grad(set_to_none=True)

            # Get training batch
            x, y = dataC.getBatch(B, T, 'train', col)
            x = x.to(device)
            y = y.to(device)

            with torch.autocast(device_type=device if device != 'cpu' else 'cpu', dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                logits, loss = model(x, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (iter + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Training iter {iter + 1}/{TRAINING_ITERS}: avg loss = {avg_loss:.4f}")

        # Save checkpoint after training
        save_checkpoint(model)

        # Switch back to eval mode
        model.eval()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"[{datetime.now()}] Training cycle complete. Final avg loss: {avg_loss:.4f}")

    except Exception as e:
        print(f"Error in training cycle: {e}")
        import traceback
        traceback.print_exc()
        # Ensure model is back in eval mode
        model.eval()


def run_prediction_cycle():
    """Run complete prediction cycle: fetch data, make prediction, verify old predictions"""
    global dataC, model

    print(f"\n[{datetime.now()}] Starting prediction cycle...")

    try:
        # 1. Pull latest USGS data
        print("Pulling USGS data...")
        dataC.usgs2DB()

        # 2. Export to CSV
        print("Exporting to CSV...")
        dataC.db2File(min_mag=3.9)

        # Reconnect after db2File closes connection
        dataC = DataC()
        dataC.getData()

        # 3. Make new prediction
        print("Making prediction...")
        make_prediction()

        # 4. Auto-verify old predictions
        print("Auto-verifying predictions...")
        auto_verify_predictions()

        print(f"[{datetime.now()}] Prediction cycle complete\n")

    except Exception as e:
        print(f"Error in prediction cycle: {e}")
        # Reconnect on error
        try:
            reconnect_data()
        except:
            pass


def reconnect_data():
    """Reconnect to database and reload data"""
    global dataC
    dataC = DataC()
    dataC.getData()


def start_scheduler():
    """Start background scheduler for automated predictions and training"""
    global scheduler

    scheduler = BackgroundScheduler()

    # Run prediction cycle every 5 minutes
    scheduler.add_job(func=run_prediction_cycle, trigger="interval", minutes=5, id='prediction_cycle')
    print("Scheduler: Prediction cycle will run every 5 minutes")

    # Run training cycle every hour
    scheduler.add_job(func=run_training_cycle, trigger="interval", hours=TRAINING_INTERVAL_HOURS, id='training_cycle')
    print(f"Scheduler: Training cycle will run every {TRAINING_INTERVAL_HOURS} hour(s)")

    scheduler.start()
    print("Scheduler started")

    # Shut down scheduler when exiting
    atexit.register(lambda: scheduler.shutdown())


# ============= API ENDPOINTS =============

@app.route('/')
def serve_index():
    """Serve the main web page"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/assets/<path:path>')
def serve_assets(path):
    """Serve static assets"""
    return send_from_directory(os.path.join(app.static_folder, 'assets'), path)


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from web folder"""
    return send_from_directory(app.static_folder, path)


@app.route('/api/live', methods=['GET'])
def get_live_data():
    """
    Get live data for dynamic display:
    - Latest prediction
    - Recent actual earthquakes
    - Success stats
    """
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        latest_prediction = dataC.get_latest_prediction()
        recent_earthquakes = dataC.get_recent_earthquakes(limit=20, min_mag=4.0)
        stats = dataC.get_prediction_stats()

        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'latest_prediction': latest_prediction,
            'recent_earthquakes': recent_earthquakes,
            'stats': stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get predictions with actual earthquakes side by side"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        limit = request.args.get('limit', 50, type=int)
        predictions = dataC.get_predictions_with_actuals(limit=limit)

        # Convert datetime objects to ISO strings
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
def refresh_data():
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
        predictions = dataC.get_predictions_with_actuals(limit=10)

        # Convert datetime objects
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
    """Get model status and configuration"""
    latest_checkpoint = get_latest_checkpoint()
    checkpoint_time = None
    if latest_checkpoint:
        checkpoint_time = datetime.fromtimestamp(os.path.getmtime(latest_checkpoint)).isoformat()

    return jsonify({
        'loaded': model is not None,
        'device': device,
        'model_type': 'ComplexEqModel',
        'latest_checkpoint': os.path.basename(latest_checkpoint) if latest_checkpoint else None,
        'checkpoint_time': checkpoint_time,
        'config': {
            'sequence_length': T,
            'embedding_size': n_embed,
            'num_heads': n_heads,
            'num_layers': n_layer,
            'prediction_target': 'multi (lat, lon, dt, mag)',
            'latitude_tolerance': LATITUDE_TOLERANCE,
            'min_mag_display': MIN_MAG_DISPLAY,
            'training_interval_hours': TRAINING_INTERVAL_HOURS,
            'training_iters_per_cycle': TRAINING_ITERS
        },
        'scheduler_running': scheduler is not None and scheduler.running if scheduler else False
    })


@app.route('/api/train', methods=['POST'])
def trigger_training():
    """Manually trigger a training cycle"""
    try:
        run_training_cycle()
        return jsonify({
            'success': True,
            'message': 'Training cycle completed',
            'checkpoint': os.path.basename(get_latest_checkpoint())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cycle', methods=['POST'])
def trigger_cycle():
    """Manually trigger a full prediction cycle"""
    try:
        run_prediction_cycle()
        return jsonify({
            'success': True,
            'message': 'Prediction cycle completed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("EARTHQUAKE PREDICTION SERVER")
    print("=" * 50)
    print("Loading model...")
    load_model()

    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt:
        print(f"Latest checkpoint: {os.path.basename(latest_ckpt)}")

    print("\nStarting scheduler...")
    start_scheduler()

    # Run initial prediction cycle
    print("\nRunning initial prediction cycle...")
    run_prediction_cycle()

    print("\n" + "=" * 50)
    print(f"Server starting on http://0.0.0.0:3000")
    print(f"Using device: {device}")
    print(f"Min magnitude display: {MIN_MAG_DISPLAY}")
    print(f"Training interval: every {TRAINING_INTERVAL_HOURS} hour(s)")
    print(f"Training iterations per cycle: {TRAINING_ITERS}")
    print("=" * 50)

    app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)
