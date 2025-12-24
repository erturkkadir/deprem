"""
Standalone Training Script for Earthquake Prediction Model
- Runs continuously
- Saves checkpoint every 200 iterations
- Pulls data from database via DataClass
"""
import os
import glob
import torch
from datetime import datetime
from DataClass import DataC
from EqModel import ComplexEqModel

# ============= CONFIGURATION =============
MODEL_DIR = '/var/www/syshuman/quake'
MODEL_PREFIX = 'eqModel_complex'
CHECKPOINT_INTERVAL = 2000  # Save every N iterations
KEEP_CHECKPOINTS = 5       # Keep last N checkpoints

# Model hyperparameters
B = 1              # Batch size
T = 1024           # Sequence length
C = 1190           # Embedding size
n_embed = C
n_heads = 8
n_layer = 8
dropout = 0.01
p_max = 180        # Prediction classes (latitude)
col = 2            # Target column (latitude)

# Training hyperparameters
LEARNING_RATE = 3e-4
RELOAD_DATA_EVERY = 1000  # Reload fresh data from DB every N iterations

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_latest_checkpoint():
    """Find the latest checkpoint file"""
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


def save_checkpoint(model, iteration, loss):
    """Save model checkpoint with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = f'{MODEL_DIR}/{MODEL_PREFIX}_{timestamp}.pth'

    # Handle torch.compile() wrapped models
    if hasattr(model, '_orig_mod'):
        state_dict = model._orig_mod.state_dict()
    else:
        state_dict = model.state_dict()

    # Clean state dict keys
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '')
        clean_state_dict[new_key] = value

    torch.save(clean_state_dict, checkpoint_path)

    # Also save to legacy path
    legacy_path = f'{MODEL_DIR}/{MODEL_PREFIX}.pth'
    torch.save(clean_state_dict, legacy_path)

    print(f"[{datetime.now()}] Checkpoint saved: {checkpoint_path} (iter={iteration}, loss={loss:.4f})")

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
                print(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                print(f"Error removing checkpoint: {e}")


def load_data():
    """Load fresh data from database"""
    print(f"[{datetime.now()}] Loading data from database...")
    dataC = DataC()
    dataC.getData()
    sizes = dataC.getSizes()
    print(f"Data loaded: {len(dataC.train):,} train, {len(dataC.valid):,} valid samples")
    return dataC, sizes


def load_model(sizes):
    """Initialize or load model from checkpoint"""
    print(f"[{datetime.now()}] Initializing model...")

    model = ComplexEqModel(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max)
    model.to(device)

    checkpoint_path = get_latest_checkpoint()

    if checkpoint_path:
        print(f"Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Clean keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    else:
        print("No checkpoint found, starting from scratch")

    return model


def train():
    """Main training loop - runs forever"""
    print("=" * 60)
    print("EARTHQUAKE PREDICTION - STANDALONE TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint interval: every {CHECKPOINT_INTERVAL} iterations")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {B}, Sequence length: {T}")
    print("=" * 60)

    # Initial data load
    dataC, sizes = load_data()

    # Load or initialize model
    model = load_model(sizes)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Use mixed precision if on CUDA
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    iteration = 0
    running_loss = 0.0
    loss_count = 0

    print(f"\n[{datetime.now()}] Starting training loop...\n")

    try:
        while True:  # Run forever
            iteration += 1

            # Reload data periodically
            if iteration % RELOAD_DATA_EVERY == 0:
                try:
                    dataC, sizes = load_data()
                except Exception as e:
                    print(f"Error reloading data: {e}, continuing with existing data")

            # Get batch
            try:
                x, y = dataC.getBatch(B, T, 'train', col)
                x = x.to(device)
                y = y.to(device)
            except Exception as e:
                print(f"Error getting batch: {e}")
                continue

            # Forward pass
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()

            # Track loss
            running_loss += loss.item()
            loss_count += 1

            # Log progress
            if iteration % 50 == 0:
                avg_loss = running_loss / loss_count
                print(f"[{datetime.now()}] Iter {iteration:,}: loss = {avg_loss:.4f}")

            # Save checkpoint
            if iteration % CHECKPOINT_INTERVAL == 0:
                avg_loss = running_loss / loss_count
                save_checkpoint(model, iteration, avg_loss)
                running_loss = 0.0
                loss_count = 0

    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Training interrupted by user")
        # Save final checkpoint
        if loss_count > 0:
            avg_loss = running_loss / loss_count
            save_checkpoint(model, iteration, avg_loss)
        print("Final checkpoint saved. Exiting.")


if __name__ == '__main__':
    train()
