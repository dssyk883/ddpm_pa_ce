import torch.nn as nn
import torch.nn.functional as F
from dataloader import MatDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import random
import sys
import os
import json
import glob
import torch
from datetime import datetime
import warnings


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (query_dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(key_dim, query_dim)
        self.to_v = nn.Linear(key_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
        
    def forward(self, x, context):
        h = self.num_heads
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Split heads
        q, k, v = map(lambda t: t.view(*t.shape[:-1], h, -1).transpose(-3, -2), (q, k, v))
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        
        # Merge heads
        out = out.transpose(-3, -2).contiguous()
        out = out.view(*out.shape[:-2], -1)
        
        return self.to_out(out)

class PilotEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Initial projection while maintaining spatial structure
        self.init_proj = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Process spatial information
        self.spatial_processor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Self-attention for pilot interactions
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x shape: [batch, 2, 18, 2]
        batch = x.shape[0]
        
        # Initial processing
        feat = self.init_proj(x)  # [batch, hidden_dim, 18, 2]
        
        # Spatial processing
        feat = self.spatial_processor(feat)
        
        # Prepare for self-attention
        feat_flat = feat.flatten(2).transpose(-1, -2)  # [batch, 36, hidden_dim]
        feat_flat = self.norm(feat_flat)
        
        # Self-attention
        feat_attn, _ = self.self_attn(feat_flat, feat_flat, feat_flat)
        feat_attn = feat_attn.transpose(-1, -2).view(batch, self.hidden_dim, 18, 2)
        
        # Residual connection
        feat = feat + feat_attn
        
        return feat

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection if channel dimensions don't match
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = F.relu(x)
        
        return x

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.mha(x, x, x)[0]
        x = x + self.ff_self(x)
        x = x.transpose(1, 2).view(-1, self.channels, *size)
        return x

class ConditionalUnet(nn.Module):
    def __init__(self, hidden_dim, in_channels=2):
        super().__init__()
        
        # Pilot encoder
        self.pilot_encoder = PilotEncoder(hidden_dim)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder
        self.enc1 = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            CrossAttention(hidden_dim, hidden_dim)
        ])
        
        self.enc2 = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            CrossAttention(hidden_dim * 2, hidden_dim)
        ])
        
        self.enc3 = nn.ModuleList([
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            CrossAttention(hidden_dim * 4, hidden_dim)
        ])
        
        # Custom downsampling
        self.downsample1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=(2, 1), padding=1)
        self.downsample2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, stride=(2, 1), padding=1)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            CrossAttention(hidden_dim * 4, hidden_dim),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1)
        ])
        
        # Decoder
        self.dec3 = nn.ModuleList([
            nn.Conv2d(hidden_dim * 8, hidden_dim * 4, 3, padding=1),
            CrossAttention(hidden_dim * 4, hidden_dim),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1)
        ])
        
        self.dec2 = nn.ModuleList([
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),
            CrossAttention(hidden_dim * 2, hidden_dim),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        ])
        
        self.dec1 = nn.ModuleList([
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            CrossAttention(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        ])
        
        self.activation = nn.GELU()
        
    def _apply_cross_attention(self, x, context, cross_attn):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(-1, -2)  # [batch, h*w, channels]
        context_flat = context.flatten(2).transpose(-1, -2)  # [batch, pilot_h*pilot_w, channels]
        
        x_attn = cross_attn(x_flat, context_flat)
        return x_attn.transpose(-1, -2).view(b, c, h, w)
    
    def forward(self, x, condition, time):
        # Process pilots through dedicated encoder
        pilot_features = self.pilot_encoder(condition)  # [batch, hidden_dim, 18, 2]
        
        # Time embedding
        t = self.time_mlp(time.unsqueeze(-1))
        t = t.view(*t.shape, 1, 1)
        
        # Encoder
        x1 = self.activation(self.enc1[0](x))
        x1 = self.activation(self.enc1[1](x1))
        x1 = x1 + self._apply_cross_attention(x1, pilot_features, self.enc1[2])
        x1 = x1 + t
        
        x2 = self.activation(self.downsample1(x1))
        x2 = self.activation(self.enc2[0](x2))
        x2 = self.activation(self.enc2[1](x2))
        x2 = x2 + self._apply_cross_attention(x2, pilot_features, self.enc2[2])
        
        x3 = self.activation(self.downsample2(x2))
        x3 = self.activation(self.enc3[0](x3))
        x3 = self.activation(self.enc3[1](x3))
        x3 = x3 + self._apply_cross_attention(x3, pilot_features, self.enc3[2])
        
        # Bottleneck
        x_mid = self.activation(self.bottleneck[0](x3))
        x_mid = x_mid + self._apply_cross_attention(x_mid, pilot_features, self.bottleneck[1])
        x_mid = self.activation(self.bottleneck[2](x_mid))
        
        # Decoder with custom upsampling
        x = self.activation(self.dec3[0](torch.cat([x_mid, x3], dim=1)))
        x = x + self._apply_cross_attention(x, pilot_features, self.dec3[1])
        x = self.activation(self.dec3[2](x))
        x = F.interpolate(x, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=True)
        
        x = self.activation(self.dec2[0](torch.cat([x, x2], dim=1)))
        x = x + self._apply_cross_attention(x, pilot_features, self.dec2[1])
        x = self.activation(self.dec2[2](x))
        x = F.interpolate(x, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)
        
        x = self.activation(self.dec1[0](torch.cat([x, x1], dim=1)))
        x = x + self._apply_cross_attention(x, pilot_features, self.dec1[1])
        x = self.dec1[2](x)
        
        return x
    
def cosine_beta_schedule(timesteps):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)


class Diffusion:
    def __init__(self, model, n_steps=1000, device="cuda"):
        self.model = model.to(device)
        self.n_steps = n_steps
        self.device = device

        # Noise schedule
        self.betas = cosine_beta_schedule(n_steps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_loss(self, x_0, condition, t):
        noise = torch.randn_like(x_0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

        # Add noise
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        # Predict noise
        pred = self.model(x_t, condition, t.float())

        # Return both MSE loss and dB loss for consistent metrics
        mse_loss = F.mse_loss(pred, noise)
        db_loss = 10 * torch.log10(mse_loss)
        return mse_loss, db_loss

    @torch.no_grad()
    def sample(self, condition, shape):
        x = torch.randn(shape).to(self.device)

        for t in reversed(range(self.n_steps)):
            t_batch = torch.tensor([t], device=self.device).repeat(shape[0])
            predicted_noise = self.model(x, condition, t_batch.float())

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise) + torch.sqrt(
                beta) * noise

        return x


def train_step(diffusion, x_0, condition, optimizer, scheduler):
    x_0, condition = x_0.to(diffusion.device), condition.to(diffusion.device)
    optimizer.zero_grad()

    # Random timesteps (0, timesteps)
    t = torch.randint(0, diffusion.n_steps, (x_0.shape[0],), device=diffusion.device)

    # Calculate loss for training - now returns both MSE and dB loss
    mse_loss, db_loss = diffusion.get_loss(x_0, condition, t)
    mse_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), max_norm=0.5)

    optimizer.step()
    scheduler.step()

    return db_loss.item()  # Return dB loss for consistent metrics


def get_sample_loss(diffusion, h_est, h_ideal):
    h_est = h_est.to(diffusion.device)
    h_ideal = h_ideal.to(diffusion.device)
    generated = diffusion.sample(h_est, shape=(h_est.shape[0], 2, 120, 14))
    mse = F.mse_loss(h_ideal, generated)
    return 10 * torch.log10(mse)


def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, output_dir, is_best=False, max_keep=5):
    """
    Save checkpoint with training state and metrics.

    Args:
        epoch (int): Current epoch number
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer state to save
        scheduler (LRScheduler): Learning rate scheduler state to save
        train_loss (float/tensor): Training loss value
        val_loss (float/tensor): Validation loss value
        output_dir (str): Directory to save checkpoints
        is_best (bool): Whether this is the best model so far
        max_keep (int): Maximum number of recent checkpoints to keep
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Helper function to convert tensors to Python native types
        def convert_tensor(value):
            if torch.is_tensor(value):
                # If it's a tensor on GPU, first move to CPU
                if value.is_cuda:
                    value = value.cpu()
                # Convert to Python number
                if value.numel() == 1:
                    return value.item()
                # If it's a multi-element tensor, convert to list
                return value.tolist()
            return value

        # Recursively convert dictionary values from tensors to native Python types
        def convert_dict(d):
            output = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    output[k] = convert_dict(v)
                elif isinstance(v, (list, tuple)):
                    output[k] = [convert_tensor(x) if torch.is_tensor(x) else x for x in v]
                else:
                    output[k] = convert_tensor(v)
            return output

        # Convert loss values to Python native types
        train_loss_value = convert_tensor(train_loss)
        val_loss_value = convert_tensor(val_loss)

        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss_value,
            'val_loss': val_loss_value,
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }

        # Save regular checkpoint
        checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)

        # Use safe save pattern
        temp_path = checkpoint_path + '.temp'
        torch.save(checkpoint, temp_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        os.rename(temp_path, checkpoint_path)

        # Save best model if this is the best performance
        if is_best:
            best_path = os.path.join(output_dir, 'best_model.pth')
            temp_best_path = best_path + '.temp'
            torch.save(checkpoint, temp_best_path)
            if os.path.exists(best_path):
                os.remove(best_path)
            os.rename(temp_best_path, best_path)

        # Manage checkpoint history
        checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint_epoch_*.pth'))
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Remove old checkpoints if exceeding max_keep
        while len(checkpoints) > max_keep:
            checkpoint_to_remove = checkpoints.pop(0)  # Remove oldest checkpoint
            try:
                os.remove(checkpoint_to_remove)
            except OSError as e:
                print(f"Warning: Could not remove old checkpoint {checkpoint_to_remove}: {e}")

        # Update metrics log
        metrics_file = os.path.join(output_dir, 'training_metrics.json')
        metrics = {'epochs': []}

        # Load existing metrics if they exist
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not read existing metrics file. Creating new one.")
                metrics = {'epochs': []}

        # Add new epoch metrics (ensuring all values are JSON serializable)
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss_value,
            'val_loss': val_loss_value,
            'is_best': is_best,
            'checkpoint_path': checkpoint_filename,
            'timestamp': checkpoint['timestamp']
        }

        # Convert any remaining tensor values to Python native types
        epoch_metrics = convert_dict(epoch_metrics)
        metrics['epochs'].append(epoch_metrics)

        # Save metrics with safe write pattern
        temp_metrics_file = metrics_file + '.temp'
        with open(temp_metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        if os.path.exists(metrics_file):
            os.remove(metrics_file)
        os.rename(temp_metrics_file, metrics_file)

        return checkpoint_path

    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        raise

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load checkpoint and return training state with better error handling."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load to CPU first

        # Verify checkpoint contents
        required_keys = ['model_state_dict', 'optimizer_state_dict',
                         'scheduler_state_dict', 'epoch', 'train_loss', 'val_loss']
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Checkpoint missing required key: {key}")

        # Load state dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

    except EOFError:
        print(f"Error: Checkpoint file {checkpoint_path} appears to be corrupted")
        raise
    except KeyError as e:
        print(f"Error: Invalid checkpoint format - {str(e)}")
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise


def verify_checkpoint(checkpoint_path):
    """Verify if checkpoint file is valid."""
    try:
        # Try to load just the metadata first
        file_size = os.path.getsize(checkpoint_path)
        if file_size == 0:
            print(f"Error: Checkpoint file {checkpoint_path} is empty")
            return False

        # Try to load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['model_state_dict', 'optimizer_state_dict',
                         'scheduler_state_dict', 'epoch', 'train_loss', 'val_loss']

        for key in required_keys:
            if key not in checkpoint:
                print(f"Error: Checkpoint missing required key: {key}")
                return False

        print(f"Checkpoint verification successful - contains data for epoch {checkpoint['epoch']}")
        return True

    except Exception as e:
        print(f"Error verifying checkpoint: {str(e)}")
        return False

def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save checkpoints and metrics')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tsteps', type=int, default=1000)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--every_n_epoch', type=int, default=10)
    parser.add_argument('--val_portion', type=float, default=1.0)
    parser.add_argument('--train_portion', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    return parser.parse_args()


if __name__ == "__main__":
    PILOT_DIMS = (18, 2)
    TRANSFORM = None
    RETURN_TYPE = "2channel"

    args = parse_params()

    # If checkpoint specified, verify it first
    if args.checkpoint:
        print(f"Verifying checkpoint: {args.checkpoint}")
        if not verify_checkpoint(args.checkpoint):
            print("Error: Invalid checkpoint file. Please check the file and try again.")
            sys.exit(1)

    config = vars(args)
    print("Started with configuration:", {k: v for k, v in config.items()})

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Device checks
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but cuda device was specified")

    if args.device.startswith('cuda'):
        device_idx = int(args.device.split(':')[1])
        if device_idx >= torch.cuda.device_count():
            raise RuntimeError(f"Specified GPU {device_idx} is not available. "
                               f"Available GPUs: {torch.cuda.device_count()}")

    # Dataset setup
    mat_dataset = MatDataset(
        data_dir=args.train_dir,
        pilot_dims=PILOT_DIMS,
        return_type=RETURN_TYPE)

    train_size = int(len(mat_dataset) * args.train_portion)
    if train_size < len(mat_dataset):
        indices = random.sample(range(len(mat_dataset)), train_size)
        mat_dataset = Subset(mat_dataset, indices)

    dataloader = DataLoader(mat_dataset, batch_size=args.batch_size, shuffle=True)

    mat_validation = MatDataset(
        data_dir=args.val_dir,
        pilot_dims=PILOT_DIMS,
        transform=None,
        return_type=RETURN_TYPE)

    val_size = int(len(mat_validation) * args.val_portion)
    if val_size < len(mat_validation):
        indices = random.sample(range(len(mat_validation)), val_size)
        mat_validation = Subset(mat_validation, indices)

    validation_dataloader = DataLoader(mat_validation, batch_size=args.batch_size, shuffle=False)

    # Model setup
    model = ConditionalUnet(hidden_dim=args.hidden, in_channels=2)
    diffusion = Diffusion(model, n_steps=args.tsteps, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

    if args.checkpoint:
        try:
            start_epoch, last_train_loss, last_val_loss = load_checkpoint(
                args.checkpoint, model, optimizer, scheduler)
            print(f"Successfully loaded checkpoint from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")
            print("Starting training from scratch...")
            start_epoch = 0
            best_val_loss = float('inf')

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} Training')

        for batch in progress_bar:
            h_est, h_ideal, _ = batch
            loss = train_step(diffusion, h_ideal, h_est, optimizer, scheduler)
            running_loss += loss
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})

        avg_train_loss = running_loss / len(dataloader)
        print(f"\nEpoch {epoch} Average Training Loss: {avg_train_loss:.4f}")
        train_loss_history.append(avg_train_loss)

        # Validation and checkpointing
        if (epoch + 1) % args.every_n_epoch == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0
            num_batch = 0

            # Added tqdm progress bar for validation
            val_progress_bar = tqdm(validation_dataloader, desc=f'Epoch {epoch} Validation')

            with torch.no_grad():
                for batch in val_progress_bar:
                    h_est, h_ideal, _ = batch
                    batch_loss = get_sample_loss(diffusion, h_est, h_ideal)
                    val_loss += batch_loss
                    num_batch += 1
                    val_progress_bar.set_postfix({'val_loss': f'{batch_loss:.4f}'})

            val_loss /= num_batch
            val_loss_history.append(val_loss)
            print(f"Validation Loss 10log(AVG_MSE): {val_loss}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Save checkpoint
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                output_dir=args.output_dir,
                is_best=is_best
            )

        scheduler.step()

    print(f"Training completed. Best validation loss: {best_val_loss}")