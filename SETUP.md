# Earthquake Prediction System - Server Setup

## Server: vanc.syshuman.com (192.168.1.177)

This server runs the AI training and prediction backend.

---

## Quick Start

```bash
cd /var/www/syshuman/quake
source venv/bin/activate

# Terminal 1: Start Training (runs forever)
python train.py

# Terminal 2: Start API Server
gunicorn --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:3000 server:app
```

---

## Components

### 1. Training Script (`train.py`)
- Runs continuously, never stops
- Saves checkpoint every 2000 iterations
- Uses GPU (CUDA) for training
- Reloads data from database every 1000 iterations

```bash
# Start training
python train.py

# Output example:
# [2025-12-23 10:00:00] Iter 50: loss = 4.2341
# [2025-12-23 10:05:00] Iter 2000: loss = 2.1234
# [2025-12-23 10:05:00] Checkpoint saved: eqModel_complex_20251223_100500.pth
```

### 2. API Server (`server.py`)
- Serves REST API on port 3000
- Uses CPU (training uses GPU)
- Makes predictions every 5 minutes
- Auto-reloads latest checkpoint from training
- Pulls USGS earthquake data
- Verifies predictions against actual earthquakes

```bash
# Start with gunicorn (recommended)
gunicorn --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:3000 server:app

# Or start directly (development)
python server.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/live` | GET | Latest prediction + recent earthquakes + stats |
| `/api/predictions` | GET | All predictions with actuals |
| `/api/predict` | POST | Trigger new prediction manually |
| `/api/refresh` | POST | Refresh USGS data |
| `/api/stats` | GET | Success rate statistics |
| `/api/model/status` | GET | Model and server status |
| `/api/reload-model` | POST | Force reload latest checkpoint |
| `/api/cycle` | POST | Run full prediction cycle |

---

## Running as Services (Production)

### Option 1: Using systemd

Create `/etc/systemd/system/quake-train.service`:
```ini
[Unit]
Description=Earthquake Prediction Training
After=network.target

[Service]
Type=simple
User=erturk
WorkingDirectory=/var/www/syshuman/quake
Environment=PATH=/var/www/syshuman/quake/venv/bin
ExecStart=/var/www/syshuman/quake/venv/bin/python train.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/quake-server.service`:
```ini
[Unit]
Description=Earthquake Prediction API Server
After=network.target

[Service]
Type=simple
User=erturk
WorkingDirectory=/var/www/syshuman/quake
Environment=PATH=/var/www/syshuman/quake/venv/bin
ExecStart=/var/www/syshuman/quake/venv/bin/gunicorn --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:3000 server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable quake-train quake-server
sudo systemctl start quake-train quake-server

# Check status
sudo systemctl status quake-train
sudo systemctl status quake-server

# View logs
sudo journalctl -u quake-train -f
sudo journalctl -u quake-server -f
```

### Option 2: Using screen/tmux

```bash
# Training
screen -S train
cd /var/www/syshuman/quake && source venv/bin/activate && python train.py
# Ctrl+A, D to detach

# Server
screen -S server
cd /var/www/syshuman/quake && source venv/bin/activate && gunicorn --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:3000 server:app
# Ctrl+A, D to detach

# Reattach
screen -r train
screen -r server
```

---

## File Structure

```
/var/www/syshuman/quake/
├── train.py              # Training script (GPU)
├── server.py             # API server (CPU)
├── EqModel.py            # Neural network model
├── DataClass.py          # Data loading & database
├── config.py             # Database credentials
├── eqModel_complex.pth   # Latest checkpoint (symlink)
├── eqModel_complex_*.pth # Timestamped checkpoints
├── data/
│   └── latest.csv        # Exported earthquake data
└── web/
    └── dist/             # Frontend build (copy to web server)
```

---

## Monitoring

### Check if services are running:
```bash
ps aux | grep -E "train.py|gunicorn"
```

### Check GPU usage (training):
```bash
nvidia-smi
```

### Test API:
```bash
curl http://localhost:3000/api/model/status
curl http://localhost:3000/api/live
```

### Check latest checkpoint:
```bash
ls -la /var/www/syshuman/quake/eqModel_complex*.pth
```

---

## Troubleshooting

### Server won't start (port in use):
```bash
sudo lsof -i :3000
sudo kill <PID>
```

### Out of GPU memory:
- Training uses GPU, server uses CPU
- Don't run both on GPU

### Database connection error:
- Check config.py credentials
- Ensure MySQL is running

### No predictions being made:
```bash
# Manually trigger prediction
curl -X POST http://localhost:3000/api/predict

# Check server logs
tail -f /tmp/server.log
```

---

## Web Server (quake.syshuman.com)

The frontend is hosted separately. It needs:

1. **Static files** from `web/dist/`
2. **api.php** proxy to forward `/api/*` to this server
3. **.htaccess** with rewrite rule

See web server for setup details.
