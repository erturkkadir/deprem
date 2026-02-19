# Earthquake Prediction System - Server Setup

## Server: vanc.syshuman.com (192.168.1.177)

This server runs the AI training and prediction backend using a **Full Complex-Valued Transformer**.

---

## Architecture Overview

### Full Complex Neural Network (272M parameters)
The model uses true complex-valued operations throughout, maintaining coupling between real and imaginary parts:

- **ComplexLinear**: Magnitude-phase parameterization `W = |W| * exp(i*θ)`
- **ComplexEmbedding**: Complex embeddings for earthquake features
- **ComplexLayerNorm/RMSNorm**: Normalization using complex variance
- **ComplexMultiHeadAttention**: Hermitian inner product with QK-Norm, PerDimScale, RoPE
- **ComplexGatedFeedForward**: SwiGLU-style with true complex multiplication
- **ResidualHead**: Output heads with MLP + linear skip connection
- **4-norm transformer blocks**: Pre-norm (LayerNorm) + Post-norm (RMSNorm) per sublayer

### Feature Encoding & Decoding Reference

**IMPORTANT**: All features are encoded to integer ranges for embedding lookup.
When making predictions, decode back to actual values using this table:

| Feature       | Actual Range     | Encoded Range | Embed Size | Encode              | Decode              |
|---------------|------------------|---------------|------------|----------------------|---------------------|
| Year          | 1970–2030        | 0–60          | 61         | `year - 1970`        | `encoded + 1970`    |
| Month         | 1–12             | 0–11          | 12         | `month - 1`          | `encoded + 1`       |
| Latitude      | -90° to +90°     | 0–180         | 181        | `lat + 90`           | `encoded - 90`      |
| Longitude     | -180° to +180°   | 0–360         | 361        | `lon + 180`          | `encoded - 180`     |
| Magnitude     | 0.0–9.1          | 0–91          | 92         | `mag * 10`           | `encoded / 10.0`    |
| Depth         | 0–200 km         | 0–200         | 201        | `depth` (clamped)    | `encoded` (km)      |
| Global dt     | 0–360 min        | 0–9           | 10         | log-binned (see below) | bin midpoint      |
| Local dt      | 0–29M min        | 0–25          | 26         | log-binned           | bin midpoint        |
| Hour          | 0–23             | 0–23          | 24         | `HOUR(datetime)`     | `encoded` (hour)    |
| Day of Year   | 0–365            | 0–365         | 366        | `DAYOFYEAR() - 1`    | `encoded + 1`       |
| Moon Phase    | 0–29             | 0–29          | 30         | orbital mechanics    | 0=new, 15=full      |
| Moon Distance | 0–9              | 0–9           | 10         | orbital mechanics    | 0=perigee, 9=apogee |

**Prediction decoding in server.py** (line ~586):
```python
lat_actual = lat_encoded - 90     # 0-180 → -90 to +90
lon_actual = lon_encoded - 180    # 0-360 → -180 to +180
mag_actual = mag_encoded / 10.0   # 0-91  → 0.0 to 9.1
```

### Global M4+ Time Difference (us_t) — Log-Binned
The `us_t` column stores raw minutes since the last global M4+ earthquake (capped at 360).
At training time, these are converted to **10 log-scale bins** matching Omori's law (aftershock rate ~ 1/t):

| Bin | Range (min) | Meaning |
|-----|-------------|---------|
| 0   | 0           | Same minute / just happened |
| 1   | 1           | Immediate triggering |
| 2   | 2-3         | Surface wave window |
| 3   | 4-7         | Dynamic stress transfer |
| 4   | 8-15        | Early aftershock peak |
| 5   | 16-31       | ~0.5 hour |
| 6   | 32-63       | ~1 hour |
| 7   | 64-127      | ~1-2 hours |
| 8   | 128-255     | ~2-4 hours |
| 9   | 256+        | >4 hours (quiet) |

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
- **Hybrid Training**: Uses M2.0+ earthquakes as input, predicts M4.0+ events
- Runs continuously with gradient accumulation (effective batch=32)
- Uses GPU (CUDA) with mixed precision training
- Cosine LR schedule with 2000-step warmup
- Saves checkpoint every 600 steps, prints every 200 steps
- Label smoothing: 0.1

```bash
# Start training
python train.py

# Or with nohup for background
nohup python train.py > train.out 2>&1 &

# Output example:
# step    200 | train 112.0784 | val 106.0108 | lr 3.00e-05
# step    400 | train 104.9742 | val 87.6003 | lr 6.00e-05
#   -> New best loss!
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
├── train.py                        # Training script (GPU)
├── server.py                       # API server (CPU)
├── EqModelComplex.py               # Full Complex neural network model
├── DataClass.py                    # Data loading & database
├── config.py                       # Database credentials
├── eqModel_complex.pth             # Latest checkpoint (symlink)
├── eqModel_complex_*.pth           # Timestamped checkpoints
├── DT.md                           # Prediction matching criteria
├── CLAUDE.md                       # AI assistant instructions
├── data/
│   └── latest.csv                  # Exported earthquake data
└── web/
    └── dist/                       # Frontend build (copy to web server)
```

### Key Model Components (EqModelComplex.py)

**Activations (Phase-Preserving)**
- `ModReLU`: `ReLU(|z| + b) * e^(i*angle(z))` - Preserves phase, learnable bias
- `ZReLU`: First quadrant only - maintains holomorphic properties
- `ComplexSiLU`: `z * sigmoid(|z|)` - Smooth gating
- `Cardioid`: Phase-dependent gating with learnable offset

**Attention Types**
- `HERMITIAN`: `Re(Q)@Re(K)^T + Im(Q)@Im(K)^T` (default, recommended)
- `REAL_PART`: `Re(Q)@Re(K)^T - Im(Q)@Im(K)^T`
- `MAGNITUDE`: `|Q @ K*|`

**Training Utilities**
- `ComplexAdamW`: Optimizer treating complex params as 2D real vectors
- `GradientClipper`: Clips gradients using complex magnitude
- `CosineWarmupScheduler`: LR warmup + cosine decay

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
- Ensure MySQL is running on 192.168.1.166

### Database Stored Procedures
The following stored procedures are used:
- `get_data_fast(min_mag)`: Fast data retrieval with pre-computed us_t
- `get_data_hybrid(input_mag, target_mag)`: Hybrid training data (M2.0+ input, M4.0+ targets)
- `ins_quakes()`: Merge staging data into main table

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

This project it to predict next earthquake. 
The idea is 
    pull data from any source either realtime or close to realtime, 
    predict next eartquake by using AI model 
    show result in web page 
    if event occurss/not occurs update succes rat on web page

The idea for AI model is this
the embedding model embeds earthquake overy year, month, x(latitude), y(longitude), m(magnitude), d(depth) and dt(time differnce between 2 earthquakes).
( I saw some pattern over month and year cycle)

    self.yr_embed = nn.Embedding(self.yr_size, n2_embed)    # 16
    self.mt_embed = nn.Embedding(self.mt_size, n2_embed)    # 16 
    self.x_embed  = nn.Embedding(self.x_size, n2_embed)     # 16
    self.y_embed  = nn.Embedding(self.y_size, n2_embed)     # 16
    self.m_embed  = nn.Embedding(self.m_size, n2_embed)     # 16
    self.d_embed  = nn.Embedding(self.d_size, n2_embed)     # 16
    self.t_embed  = nn.Embedding(self.t_size, n2_embed)     # 96
Use complex valued Embedding and attention layers

web folder is to represent all result
data folder is to save trainig data
database is yysql and connection paramters are in config.py file
pulling data from usgs code and save is at database class

WHEN CODE CHANGES UPDATE THIS FILE AND "How It Works" SECTION ACCORDINGLY

