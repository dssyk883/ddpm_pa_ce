import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import MatDataset
from torch.utils.data import DataLoader
import argparse


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
        # self.betas = torch.linspace(1e-4, 0.02, n_steps).to(device) # Original
        self.betas = cosine_beta_schedule(n_steps).to(device) # Cosine beta scheduler s=0.008
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_loss(self, x_0, condition, t):
        noise = torch.randn_like(x_0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        
        # Add noise
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        
        # Predict noise
        pred = self.model(x_t, condition, t.float())
        
        return F.mse_loss(pred, noise)

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
                
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise) + torch.sqrt(beta) * noise
            
        return x

# Training
def train_step(diffusion, x_0, condition, optimizer):
    x_0, condition = x_0.to(diffusion.device), condition.to(diffusion.device)
    optimizer.zero_grad()
    
    # Random timesteps (0, timesteps)
    t = torch.randint(0, diffusion.n_steps, (x_0.shape[0],), device=diffusion.device)
    
    # Calculate loss for training
    
    # with torch.amp.autocast(device_type='cuda'):
    #     loss = diffusion.get_loss(x_0, condition, t)

    loss = diffusion.get_loss(x_0, condition, t)
    loss.backward()
    optimizer.step()
    torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), max_norm=0.5)

    # scheduler.step()

    return loss.item()

def get_sample_loss(diffusion, h_est, h_ideal):
    h_est = h_est.to(diffusion.device)
    h_ideal = h_ideal.to(diffusion.device)
    generated = diffusion.sample(h_est, shape=(h_est.shape[0], 2, 120, 14))
    mse = F.mse_loss(h_ideal, generated)
    return 10 * torch.log10(mse)

PILOT_DIMS = (18, 2)
TRANSFORM = None
RETURN_TYPE = "2channel"

def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tsteps', type=int, default=1000)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0', 
                      help='Device to run on (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--every_n_epoch', type=int, default=10,
                      help='Run validation every n epochs')
    parser.add_argument('--val_portion', type=float, default=1.0,
                      help='Portion of validation set to use (0.0-1.0)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_params()
    config = vars(args)
    print("Started with configuration:", 
    {k: v for k, v in config.items() if k in ['data_dir', 'batch_size', 'tsteps', 'hidden', 'lr', 'epochs', 'device']})

    # Check if CUDA is available when cuda device is specified
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but cuda device was specified")

    # If cuda device is specified, verify it exists
    if args.device.startswith('cuda'):
        device_idx = int(args.device.split(':')[1])
        if device_idx >= torch.cuda.device_count():
            raise RuntimeError(f"Specified GPU {device_idx} is not available. "
                             f"Available GPUs: {torch.cuda.device_count()}")

    mat_dataset = MatDataset(
        data_dir=args.train_dir,
        pilot_dims=PILOT_DIMS,
        transform=None,
        return_type=RETURN_TYPE)

    dataloader = DataLoader(mat_dataset, batch_size=args.batch_size, shuffle=True)

    mat_validation = MatDataset(
        data_dir=args.val_dir,
        pilot_dims=PILOT_DIMS,
        transform=None,
        return_type=RETURN_TYPE
    )
    
    # Calculate validation subset size
    val_size = int(len(mat_validation) * args.val_portion)
    if val_size < len(mat_validation):
        from torch.utils.data import Subset
        import random
        indices = random.sample(range(len(mat_validation)), val_size)
        mat_validation = Subset(mat_validation, indices)

    validation_dataloader = DataLoader(mat_validation, batch_size=args.batch_size, shuffle=False)

    mat_testset = MatDataset(
        data_dir=args.test_dir,
        pilot_dims=PILOT_DIMS,
        transform=None,
        return_type=RETURN_TYPE)

    test_dataloader = DataLoader(mat_testset, batch_size=args.batch_size, shuffle=False)

    model = ConditionalUnet(hidden_dim=args.hidden, in_channels=2)
    diffusion = Diffusion(model, n_steps=args.tsteps, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=1e-6
    )

    # total_steps = len(dataloader) * args.epochs

    # Initialize scheduler with warmup
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=args.lr,
    #     total_steps=len(dataloader) * args.epochs,
    #     pct_start=0.3,  # 10% of total steps for warmup
    #     div_factor=25,  # initial_lr = max_lr/25
    #     final_div_factor=1e4,  # min_lr = initial_lr/1000
    #     )

    scaler = torch.amp.GradScaler()

    best_val_loss = float('inf')
    num_epochs = args.epochs
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        for batch in dataloader:  
            h_est, h_ideal, _ = batch
            loss = train_step(diffusion, h_ideal, h_est, optimizer)
            print(f"Epoch {epoch}, Train Loss: {loss}")
        
        # Only run validation every n epochs or on the final epoch
        if (epoch + 1) % args.every_n_epoch == 0 or epoch == num_epochs - 1:
            model.eval()
            val_loss = 0
            num_batch = 0
            with torch.no_grad():
                for batch in validation_dataloader:
                    h_est, h_ideal, _ = batch
                    val_loss += get_sample_loss(diffusion, h_est, h_ideal)
                    num_batch += 1
            val_loss /= num_batch
            print(f"Validation Loss 10log(AVG_MSE): {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 3:
                print("Early Stopping")
                break
        
        scheduler.step()
        # torch.cuda.empty_cache()
        # gc.collect()
