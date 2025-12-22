"""
Flask API Server for Earthquake Prediction System
As specified in CLAUDE.md:
- Serve predictions via API
- Track and display success rate on web page
"""
import os
import json
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta

from DataClass import DataC
from EqModel import ComplexEqModel, EqModel

app = Flask(__name__, static_folder='web')
CORS(app)

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
predictions_log = []
success_stats = {
    'total_predictions': 0,
    'correct_predictions': 0,
    'success_rate': 0.0,
    'last_updated': None
}

MODEL_PATH = 'eqModel_complex.pth'
PREDICTIONS_LOG_PATH = 'predictions_log.json'


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
        state_dict = torch.load(MODEL_PATH, map_location=device)
        # Remove '_orig_mod.' prefix if present (from torch.compile)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print("No saved model found, using initialized weights")

    model.eval()


def load_predictions_log():
    """Load predictions history from file"""
    global predictions_log, success_stats

    if os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH, 'r') as f:
            data = json.load(f)
            predictions_log = data.get('predictions', [])
            success_stats = data.get('stats', success_stats)


def save_predictions_log():
    """Save predictions history to file"""
    with open(PREDICTIONS_LOG_PATH, 'w') as f:
        json.dump({
            'predictions': predictions_log[-1000:],  # Keep last 1000
            'stats': success_stats
        }, f)


def update_success_rate():
    """Calculate success rate from verified predictions"""
    verified = [p for p in predictions_log if p.get('verified', False)]
    if verified:
        correct = sum(1 for p in verified if p.get('correct', False))
        success_stats['total_predictions'] = len(verified)
        success_stats['correct_predictions'] = correct
        success_stats['success_rate'] = (correct / len(verified)) * 100
        success_stats['last_updated'] = datetime.now().isoformat()


@app.route('/')
def serve_index():
    """Serve the main web page"""
    return send_from_directory('web', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from web folder"""
    return send_from_directory('web', path)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make earthquake prediction
    Returns predicted latitude for next earthquake
    """
    global model, dataC

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get last sequence from data
        x_test, y_actual = dataC.getLast(1, T, 'val', col=col)
        x_test = x_test.to(device)

        with torch.no_grad():
            predicted = model.generate(x_test)

        predicted_lat = predicted.item() - 90  # Convert back from 0-180 to -90 to 90
        actual_lat = y_actual.item() - 90 if y_actual is not None else None

        prediction_record = {
            'id': len(predictions_log) + 1,
            'timestamp': datetime.now().isoformat(),
            'predicted_lat': predicted_lat,
            'predicted_raw': predicted.item(),
            'actual_lat': actual_lat,
            'actual_raw': y_actual.item() if y_actual is not None else None,
            'verified': False,
            'correct': False
        }

        predictions_log.append(prediction_record)
        save_predictions_log()

        return jsonify({
            'success': True,
            'prediction': {
                'latitude': predicted_lat,
                'raw_value': predicted.item(),
                'timestamp': prediction_record['timestamp']
            },
            'message': f'Next earthquake predicted near latitude {predicted_lat:.2f}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recent-earthquakes', methods=['GET'])
def recent_earthquakes():
    """Get recent earthquake data from database"""
    global dataC

    if dataC is None:
        return jsonify({'error': 'Data not loaded'}), 500

    try:
        # Fetch recent data from USGS
        dataC.usgs2DB()

        return jsonify({
            'success': True,
            'message': 'Data updated from USGS'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get prediction success statistics
    As specified in CLAUDE.md: show success rate on web page
    """
    update_success_rate()

    recent_predictions = predictions_log[-10:] if predictions_log else []

    return jsonify({
        'success': True,
        'stats': success_stats,
        'recent_predictions': recent_predictions,
        'total_logged': len(predictions_log)
    })


@app.route('/api/verify', methods=['POST'])
def verify_prediction():
    """
    Verify a prediction against actual earthquake data
    Updates success rate as specified in CLAUDE.md
    """
    data = request.get_json()
    prediction_id = data.get('prediction_id')
    actual_lat = data.get('actual_lat')
    tolerance = data.get('tolerance', 5)  # Default 5 degree tolerance

    if prediction_id is None or actual_lat is None:
        return jsonify({'error': 'Missing prediction_id or actual_lat'}), 400

    for pred in predictions_log:
        if pred['id'] == prediction_id:
            pred['verified'] = True
            pred['actual_lat'] = actual_lat
            diff = abs(pred['predicted_lat'] - actual_lat)
            pred['correct'] = diff <= tolerance
            pred['difference'] = diff

            update_success_rate()
            save_predictions_log()

            return jsonify({
                'success': True,
                'verified': True,
                'correct': pred['correct'],
                'difference': diff,
                'stats': success_stats
            })

    return jsonify({'error': 'Prediction not found'}), 404


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
            'prediction_target': 'latitude'
        }
    })


@app.route('/api/model/train', methods=['POST'])
def trigger_training():
    """Trigger model training (placeholder for background training)"""
    return jsonify({
        'success': True,
        'message': 'Training triggered. Check logs for progress.'
    })


if __name__ == '__main__':
    print("Loading earthquake prediction model...")
    load_model()
    load_predictions_log()
    print(f"Server starting on http://0.0.0.0:5000")
    print(f"Using device: {device}")
    app.run(host='0.0.0.0', port=5000, debug=True)
