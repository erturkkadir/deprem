"""
Earthquake Prediction Model Training Script
Uses ComplexEqModel with complex-valued embeddings and attention
Hybrid approach: M2+ input context, M4+ targets
"""
import torch
from DataClass import DataC
from EqModel import ComplexEqModel
import numpy as np
from datetime import datetime
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train earthquake prediction model')
parser.add_argument('--epochs', type=int, default=50000, help='Number of training iterations')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
parser.add_argument('--input-mag', type=float, default=2.0, help='Min magnitude for input (default: 2.0)')
parser.add_argument('--target-mag', type=float, default=4.0, help='Min magnitude for targets (default: 4.0)')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
args = parser.parse_args()

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
B = args.batch_size
T = args.seq_len
n_embed = 1176          # embedding size (7 features * 168 each, divisible by 8 heads)
n_heads = 8             # attention heads
n_layer = 8             # transformer blocks
dropout = 0.01
lr = args.lr
max_iters = args.epochs
eval_interval = 2000
save_interval = 5000

print("\n" + "=" * 60)
print("EARTHQUAKE PREDICTION MODEL - HYBRID TRAINING")
print("=" * 60)

# Load hybrid data
print(f"\nLoading data (M{args.input_mag}+ input, M{args.target_mag}+ targets)...")
dataC = DataC()
data = dataC.getDataHybrid(input_mag=args.input_mag, target_mag=args.target_mag)
sizes = dataC.getSizes()

# Test batch
x, targets = dataC.getBatchHybrid(B, T, 'train')

print(f"\nModel Configuration:")
print(f"  Device      : {device}")
print(f"  Batch size  : {B}")
print(f"  Seq length  : {T}")
print(f"  n_embed     : {n_embed:,}")
print(f"  n_heads     : {n_heads}")
print(f"  n_layers    : {n_layer}")
print(f"  dropout     : {dropout}")
print(f"  lr          : {lr}")
print(f"  max_iters   : {max_iters:,}")
print(f"  Input shape : {x.shape}")
print("=" * 60)

# Create model
model = ComplexEqModel(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max=181)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# Resume from checkpoint if specified
start_iter = 0
if args.resume and os.path.exists(args.resume):
    print(f"Resuming from {args.resume}")
    model.load_state_dict(torch.load(args.resume, map_location=device))

# Note: torch.compile disabled for complex-valued model (not supported well)
print("Using eager mode (complex tensors don't support torch.compile well)")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Loss tracking
loss_history = []
best_val_loss = float('inf')

@torch.no_grad()
def estimate_loss(num_batches=50):
    """Estimate loss on train and validation sets."""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(num_batches):
            try:
                x, targets = dataC.getBatchHybrid(B, T, split)
                x = x.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                _, loss = model(x, targets)
                losses.append(loss.item())
            except Exception as e:
                continue
        out[split] = np.mean(losses) if losses else float('inf')
    model.train()
    return out

@torch.no_grad()
def test_predictions(num_samples=10):
    """Test model predictions on random samples."""
    model.eval()
    results = []
    for _ in range(num_samples):
        x, targets = dataC.getBatchHybrid(1, T, 'val')
        x = x.to(device)
        pred = model.generate(x)
        results.append({
            'pred_lat': pred['lat'] - 90,
            'pred_lon': pred['lon'] - 180,
            'pred_mag': pred['mag'] / 10,
            'pred_dt': pred['dt'],
            'true_lat': targets['lat'].item() - 90,
            'true_lon': targets['lon'].item() - 180,
            'true_mag': targets['mag'].item() / 10,
            'true_dt': targets['dt'].item(),
        })
    model.train()
    return results

def save_checkpoint(model, suffix='', keep_last=5):
    """Save model checkpoint with timestamp, delete old ones."""
    import glob

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'eqModel_complex_{timestamp}{suffix}.pth'
    torch.save(model.state_dict(), filename)

    # Delete old checkpoints, keeping only the last N
    pattern = f'eqModel_complex_*{suffix}.pth'
    checkpoints = sorted(glob.glob(pattern))
    if len(checkpoints) > keep_last:
        for old_file in checkpoints[:-keep_last]:
            try:
                os.remove(old_file)
            except:
                pass

    return filename

# Training loop
print("\n" + "." * 60)
print("TRAINING STARTED...")
print("." * 60 + "\n")

model.train()

for iter in range(max_iters):
    # Get batch
    try:
        x, targets = dataC.getBatchHybrid(B, T, 'train')
    except Exception as e:
        print(f"Batch error: {e}")
        continue

    x = x.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}

    # Forward pass (no autocast for complex tensors - they don't support bfloat16)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, targets)

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    # Track loss
    loss_history.append(loss.item())

    # Evaluation
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        avg_loss = np.mean(loss_history[-eval_interval:]) if loss_history else 0

        print(f"step {iter:5d} | loss {avg_loss:.4f} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        # Save best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            filename = save_checkpoint(model, '_best')
            print(f"  -> Best model saved: {filename}")

    # Periodic checkpoint
    if iter > 0 and iter % save_interval == 0:
        filename = save_checkpoint(model)
        print(f"  -> Checkpoint saved: {filename}")

# Final save
filename = save_checkpoint(model, '_final')
print(f"\nTraining complete. Final model: {filename}")

# Test predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (Validation Set)")
print("=" * 60)

results = test_predictions(10)
for i, r in enumerate(results):
    lat_diff = abs(r['pred_lat'] - r['true_lat'])
    lon_diff = abs(r['pred_lon'] - r['true_lon'])
    print(f"[{i+1:2d}] Pred: ({r['pred_lat']:+6.1f}, {r['pred_lon']:+7.1f}) M{r['pred_mag']:.1f} +{r['pred_dt']:3d}min")
    print(f"     True: ({r['true_lat']:+6.1f}, {r['true_lon']:+7.1f}) M{r['true_mag']:.1f} +{r['true_dt']:3d}min")
    print(f"     Diff: lat={lat_diff:.1f}° lon={lon_diff:.1f}°")
    print()

print("=" * 60)
