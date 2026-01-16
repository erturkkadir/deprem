"""
Earthquake Prediction Model - Hybrid Training Script
- Uses M2+ input context, M4+ targets
- Runs continuously
- Prints loss every 2000 steps
- Saves checkpoint every 5000 steps
"""
import os
import sys
import glob
import json
import torch
from datetime import datetime
from DataClass import DataC
from EqModelComplex import EqModelComplex

# Force unbuffered output for real-time monitoring
sys.stdout.reconfigure(line_buffering=True)

# ============= CONFIGURATION =============
MODEL_DIR = '/var/www/syshuman/quake'
MODEL_PREFIX = 'eqModel_complex'
TRAINING_STATUS_FILE = f'{MODEL_DIR}/training_status.json'
PRINT_INTERVAL = 200       # Print loss every N iterations
CHECKPOINT_INTERVAL = 600  # Save every N iterations
KEEP_CHECKPOINTS = 5       # Keep last N checkpoints

# Model hyperparameters
B = 2              # Batch size (reduced for memory)
T = 256            # Sequence length (reduced for memory)
n_embed = 1176     # Embedding size (divisible by 7 features and 8 heads)
n_heads = 8
n_layer = 6        # Reduced layers for memory
dropout = 0.01

# Training hyperparameters
LEARNING_RATE = 3e-4
INPUT_MAG = 2.0    # Min magnitude for input context
TARGET_MAG = 4.0   # Min magnitude for targets

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_latest_checkpoint():
    """Find the latest checkpoint file"""
    pattern = f'{MODEL_DIR}/{MODEL_PREFIX}_*.pth'
    checkpoints = glob.glob(pattern)

    if checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]

    return None


def update_training_status(iteration, loss, is_checkpoint=False):
    """Write training status to JSON file for server to read"""
    try:
        status = {
            'latest_step': iteration,
            'latest_loss': loss,
            'updated_at': datetime.now().isoformat()
        }
        if is_checkpoint:
            status['checkpoint_step'] = iteration
            status['checkpoint_loss'] = loss

        # Try to preserve checkpoint info from previous status
        if os.path.exists(TRAINING_STATUS_FILE) and not is_checkpoint:
            try:
                with open(TRAINING_STATUS_FILE, 'r') as f:
                    old_status = json.load(f)
                    if 'checkpoint_step' in old_status:
                        status['checkpoint_step'] = old_status['checkpoint_step']
                        status['checkpoint_loss'] = old_status['checkpoint_loss']
            except:
                pass

        with open(TRAINING_STATUS_FILE, 'w') as f:
            json.dump(status, f)
    except Exception as e:
        print(f"Warning: Could not update training status: {e}")


def save_checkpoint(model, iteration, loss):
    """Save model checkpoint with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = f'{MODEL_DIR}/{MODEL_PREFIX}_{timestamp}.pth'

    torch.save(model.state_dict(), checkpoint_path)

    # Also save to standard path for server
    standard_path = f'{MODEL_DIR}/{MODEL_PREFIX}.pth'
    torch.save(model.state_dict(), standard_path)

    print(f"[{datetime.now()}] Checkpoint saved: {checkpoint_path} (iter={iteration}, loss={loss:.4f})")

    # Update training status file with checkpoint info
    update_training_status(iteration, loss, is_checkpoint=True)

    # Cleanup old checkpoints
    cleanup_old_checkpoints()

    return checkpoint_path


def cleanup_old_checkpoints():
    """Remove old checkpoints, keeping only the most recent ones"""
    pattern = f'{MODEL_DIR}/{MODEL_PREFIX}_*.pth'
    checkpoints = glob.glob(pattern)

    if len(checkpoints) > KEEP_CHECKPOINTS:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        for old_checkpoint in checkpoints[KEEP_CHECKPOINTS:]:
            try:
                os.remove(old_checkpoint)
            except:
                pass


def load_data():
    """Load hybrid data from database"""
    print(f"[{datetime.now()}] Loading hybrid data (M{INPUT_MAG}+ input, M{TARGET_MAG}+ targets)...")
    dataC = DataC()
    dataC.getDataHybrid(input_mag=INPUT_MAG, target_mag=TARGET_MAG)
    sizes = dataC.getSizes()
    return dataC, sizes


def train():
    """Main training loop"""
    print("=" * 60)
    print("EARTHQUAKE PREDICTION - HYBRID TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input: M{INPUT_MAG}+  |  Targets: M{TARGET_MAG}+")
    print(f"Batch: {B}  |  Seq len: {T}  |  Embed: {n_embed}")
    print(f"Print every: {PRINT_INTERVAL}  |  Save every: {CHECKPOINT_INTERVAL}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 60)

    # Load data
    dataC, sizes = load_data()

    # Create model with Geographic Positional Encoding (GPE) for enhanced location awareness
    # GPE includes: Spherical Harmonics, Fourier Features, Tectonic Zones, Relative Position
    print(f"\n[{datetime.now()}] Creating model with Geographic Positional Encoding (GPE)...")
    model = EqModelComplex(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max=181, use_rope=True, use_gpe=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load checkpoint if exists and compatible
    # NOTE: GPE architecture change requires fresh start - old checkpoints won't be compatible
    checkpoint_path = get_latest_checkpoint()
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully")
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key" in str(e):
                print(f"Checkpoint incompatible with new GPE architecture, starting fresh")
            else:
                raise e
    else:
        print("Starting from scratch with GPE architecture")

    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    iteration = 0
    running_loss = 0.0
    best_loss = float('inf')

    print(f"\n[{datetime.now()}] Training started...\n")

    try:
        while True:
            iteration += 1

            # Get batch
            try:
                x, targets = dataC.getBatchHybrid(B, T, 'train')
                x = x.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
            except Exception as e:
                print(f"Batch error: {e}")
                continue

            # Forward pass (no autocast - complex tensors don't support bfloat16)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

            # Print progress
            if iteration % PRINT_INTERVAL == 0:
                avg_loss = running_loss / PRINT_INTERVAL
                print(f"step {iteration:6d} | loss {avg_loss:.4f}")

                # Update status file for server
                update_training_status(iteration, avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    print(f"  -> New best loss!")

                running_loss = 0.0

            # Save checkpoint
            if iteration % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(model, iteration, avg_loss)

    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Interrupted by user")
        if running_loss > 0:
            avg_loss = running_loss / (iteration % PRINT_INTERVAL or 1)
            save_checkpoint(model, iteration, avg_loss)
        print("Checkpoint saved. Exiting.")


if __name__ == '__main__':
    train()
