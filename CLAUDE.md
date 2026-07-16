# Earthquake Prediction System - Server Setup

## Server: vanc.syshuman.com (192.168.1.177)

This server runs the AI training and prediction backend using a **Full Complex-Valued Transformer**.

---

## Architecture Overview

### Full Complex Neural Network (203M parameters)
Current config: n_layer=6, n_embed=1024, n_heads=8, T=384. Shares GPU with Ollama.
The model uses true complex-valued operations throughout, maintaining coupling between real and imaginary parts:

- **ComplexLinear**: Magnitude-phase parameterization `W = |W| * exp(i*θ)`
- **ComplexEmbedding**: Complex embeddings for earthquake features
- **ComplexLayerNorm/RMSNorm**: Normalization using complex variance
- **ComplexMultiHeadAttention**: Hermitian inner product with QK-Norm, PerDimScale, RoPE
- **ComplexGatedFeedForward**: SwiGLU-style with true complex multiplication
- **SpatialMDNHead**: K=20 bivariate Gaussian mixture for joint (lat, lon) prediction + diversity_loss (τ=30°)
- **MagnitudeMDNHead**: K=8 univariate Gaussian mixture (GR prior REMOVED — was suppressing M>4.43)
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

**Prediction output**: MDN `generate()` returns actual coordinates directly:
```python
lat_actual = predictions['lat']   # -90 to +90 (float)
lon_actual = predictions['lon']   # -180 to +180 (float)
mag_actual = predictions['mag']   # 2.0 to 9.5 (float)
# Encode for DB storage:
lat_encoded = int(round(lat_actual + 90))
lon_encoded = int(round(lon_actual + 180))
mag_encoded = int(round(mag_actual * 10))
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

### Binary Forecast Accounting + E. Mediterranean Region (v1.6, 2026-07-14)
Supersedes the alerts-only headline of v1.5. **"YA BİLDİK YA BİLEMEDİK" +
LATE CATCH** (final form, 2026-07-16 — user: only earthquakes matter; no
quiet credit, no false alarms; an early prediction fulfilled late = late catch):
- M4+ within **250 km** of the prediction, inside the 60-min window →
  `caught` ("Eşleşti", pr_event_occurred=1, pr_correct=1)
- Same but within **LATE_CATCH_HOURS=48** after the window → `late`
  ("Geç Yakalama", pr_event_occurred=2, pr_correct=1, still a success;
  claimed pr_actual_ids prevent one quake crediting multiple cycles)
- In-window region M4+ too far, never followed by a near one → `missed_event`
  (pr_event_occurred=1, pr_correct=0; upgradable to late for 48h)
- Nothing relevant → pr_event_occurred=0, NOT scored, hidden from the list
- **NO false-alarm concept**; the alert gate (p_event>=0.90) only governs emails
**Headline = event_success = (caught + late) / graded events** ('—' until the
first event). A far in-window quake does NOT close the cycle early — a near
one may still arrive before the window ends. `auto_verify_predictions()`
(RECHECK_HOURS=50): in-window catalog-latency recheck → caught; late-catch
upgrade (missed→late allowed); orphaned unverified cycles finalized +10 min.
Stats keys: success_rate==event_success, events_caught, late_catches,
events_missed, events_occurred, alerts. UI tiles: Deprem İsabeti / Eşleşen /
Geç Yakalama / Kaçırılan; list filters All/Pending/Matched/Late/Missed;
API per-row `outcome`: pending|caught|late|missed_event|quiet_ok(hidden).
NOTE (user decision 2026-07-16): do NOT go global — global M4+ rate 44.8/day
means ~85% of windows have an event somewhere and per-cycle 250-km matching
would collapse the metric; regional focus is what makes high success attainable.

**Region** (expanded 2026-07-14): Eastern Mediterranean, lat 30–45 N /
lon 19–50 E (encoded 120–135 / 199–230) — Turkey + Greece + Aegean + Hellenic
arc + Cyprus + Caucasus edge. Training targets 17,959 (was 8,351 Turkey-only);
occ label base rate 3.89% (was 1.83%). Same bbox everywhere: DataClass target
mask + occ labels, model bbox-containment loss + component init, server
REGION_* constants.

**Training labels** (`DataClass.getDataHybrid`):
- Direct SQL adds `UNIX_TIMESTAMP` as col 13 → 14 cols:
  `[yr,mo,x,y,m,d,dt,lt,hour,doy,moon_phase,moon_dist,is_target,unix_ts]`
- `self.occ_label[i]` = 1 if any region M4+ occurs within 60 min strictly after
  event i (searchsorted, leak-free). `targets['occ']` [B,T] dense at every position.

**Model** (`EqModelComplex.py`):
- `OccurrenceHead(n_embed*2 → 512 → 1)`, plain BCE (calibrated), weight 2.0
- bbox containment loss clamps to lat[30,45]/lon[19,50]; MDN init includes
  Corinth, Ionian, Crete, Cyprus, Caucasus zones

**Best-checkpoint serving (critical)**:
- `train.py` writes `eqModel_complex_best.pth` (atomic os.replace) whenever
  val loss improves; excluded from cleanup and from training resume
- `server.py get_latest_checkpoint()` prefers `_best.pth`; reload detection
  uses mtime (path never changes). NEVER serve the latest checkpoint — v1.5
  served latest, model memorized train set (train -0.24 / val 7.0) and emitted
  avg p_event 0.44 vs true ~2% → 24 false alarms in 11 days, 0% precision.
- Anti-overfit: LR 3e-5 (was 1e-4), weight_decay 0.05 (was 0.01)

**Server verification** (`server.py`):
- `finalize_cycle(pr_id, event_occurred, is_alert, actual)` in DataClass sets
  pr_verified/pr_event_occurred/pr_correct (+ closest actual for the map)
- Event during window → finalize immediately (TP if alert, FN if monitor) → new cycle
- Window expires quietly → TN if monitor, FP if alert → new cycle
- `auto_verify_predictions()` re-grades cycles from last `RECHECK_HOURS=24` as
  late catalog data arrives (USGS/EMSC latency); old 48h late-catch removed
- DB: `pr_event_occurred TINYINT` column (migration in `_ensure_prediction_columns`)

**Stats** (`get_prediction_stats`): headline `success_rate` = accuracy;
plus `alerts, alerts_correct, false_alarms, alert_precision, events_occurred,
events_caught, events_missed, event_recall` (+ legacy `all_*` keys).

**UI**: stats row = Accuracy / Correct Forecasts / Events Caught / Missed
Events / False Alarms / Alert Precision (LiveDashboard + StatsGrid, en/tr/ja).

**Calibration note**: after training converges, verify empirically that among
validation positions with p>=0.9 the hit rate is >=90%; adjust ALERT_THRESHOLD
if BCE calibration drifts. Old v1.5 checkpoints archived in `old_checkpoints_v15/`.

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

