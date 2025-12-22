"""
Flask API Server for Earthquake Prediction System
- Automated USGS data pull every 5 minutes
- Auto-verification of predictions against actual earthquakes
- Serves API on port 1977 for external access
"""
import os
import json
import torch
import atexit
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from DataClass import DataC
from EqModel import ComplexEqModel, EqModel

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

MODEL_PATH = 'eqModel_complex.pth'
LATITUDE_TOLERANCE = 10  # degrees


def load_model():
    """Load or initialize the earthquake prediction model"""
    global model, dataC

    dataC = DataC()
    dataC.getData()
    sizes = dataC.getSizes()

    # Use ComplexEqModel as specified in CLAUDE.md
    model = ComplexEqModel(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max)
    model.to(device)

    if os.path.exists(MODEL_PATH):
        # Load state dict and handle torch.compile() prefix
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
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
        print(f"Model loaded from {MODEL_PATH}")
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

        # Save prediction to database
        pr_id = dataC.save_prediction(lat_encoded, lon_encoded, dt_minutes, mag_encoded)

        lat_actual = lat_encoded - 90
        lon_actual = lon_encoded - 180
        mag_actual = mag_encoded / 10.0

        print(f"[{datetime.now()}] Prediction made: lat={lat_actual}°, lon={lon_actual}°, dt={dt_minutes}min, mag={mag_actual} (id={pr_id})")
        return pr_id

    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


LONGITUDE_TOLERANCE = 20  # degrees for longitude
DT_TOLERANCE = 30  # minutes for time difference
MAG_TOLERANCE = 1.0  # magnitude difference

def auto_verify_predictions():
    """Auto-verify predictions against actual earthquakes"""
    global dataC

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
                # Find the earthquake with closest combined score (weighted by lat, lon)
                best_match = None
                min_score = float('inf')

                for actual in actuals:
                    us_id, us_datetime, us_x, us_y, us_m, us_mag, us_place = actual
                    # Calculate differences
                    lat_diff = abs(pr_lat_predicted - us_x) if pr_lat_predicted else 180
                    lon_diff = abs(pr_lon_predicted - us_y) if pr_lon_predicted else 360
                    # Handle longitude wrap-around
                    lon_diff = min(lon_diff, 360 - lon_diff)

                    # Combined score (weighted)
                    score = lat_diff + (lon_diff * 0.5)  # Weight lat more heavily
                    if score < min_score:
                        min_score = score
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

                    # Determine correctness: lat within tolerance AND magnitude within tolerance
                    lat_correct = diff_lat <= LATITUDE_TOLERANCE if diff_lat else False
                    lon_correct = diff_lon <= LONGITUDE_TOLERANCE if diff_lon else True
                    mag_correct = diff_mag <= MAG_TOLERANCE if diff_mag else True

                    correct = lat_correct and lon_correct

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
                    print(f"[{datetime.now()}] Verified prediction {pr_id}: lat_diff={diff_lat}°, lon_diff={diff_lon}°, mag_diff={diff_mag}, correct={correct}")

    except Exception as e:
        print(f"Error in auto-verification: {e}")
        import traceback
        traceback.print_exc()


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
    """Start background scheduler for automated predictions"""
    global scheduler

    scheduler = BackgroundScheduler()
    # Run prediction cycle every 5 minutes
    scheduler.add_job(func=run_prediction_cycle, trigger="interval", minutes=5, id='prediction_cycle')
    scheduler.start()
    print("Scheduler started - prediction cycle will run every 5 minutes")

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
    return jsonify({
        'loaded': model is not None,
        'device': device,
        'model_type': 'ComplexEqModel',
        'config': {
            'sequence_length': T,
            'embedding_size': n_embed,
            'num_heads': n_heads,
            'num_layers': n_layer,
            'prediction_target': 'latitude',
            'latitude_tolerance': LATITUDE_TOLERANCE
        },
        'scheduler_running': scheduler is not None and scheduler.running if scheduler else False
    })


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

    print("Starting scheduler...")
    start_scheduler()

    # Run initial prediction cycle
    print("Running initial prediction cycle...")
    run_prediction_cycle()

    print(f"\nServer starting on http://0.0.0.0:1977")
    print(f"Using device: {device}")
    print("=" * 50)

    app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)
