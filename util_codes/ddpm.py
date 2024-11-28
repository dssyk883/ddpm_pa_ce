import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import MatDataset
from torch.utils.data import DataLoader
import argparse
import gc

class ConditionalUNet(nn.Module):
    def __init__(self, hidden_dim, in_channels=2):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1)
	
        # Decoder
        self.dec1 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        self.dec2 = nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        
        # Condition projection
        self.cond_proj = nn.Conv2d(2, hidden_dim, 1)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, condition, time):
        # Estimates (condition)
        cond = self.cond_proj(condition)
        cond = F.interpolate(cond, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1 + cond))
        
        # Decoder
        x = F.relu(self.dec1(x2))
        x = self.dec2(x + cond)
        
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
        # self.betas = torch.linspace(0.0001, 0.02, n_steps).to(device) # Original
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
    loss = diffusion.get_loss(x_0, condition, t)
    loss.backward()
    optimizer.step()
    
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

    validation_dataloader = DataLoader(mat_validation, batch_size=args.batch_size, shuffle=False)

    mat_testset = MatDataset(
        data_dir=args.test_dir,
        pilot_dims=PILOT_DIMS,
        transform=None,
        return_type=RETURN_TYPE)

    test_dataloader = DataLoader(mat_testset, batch_size=args.batch_size, shuffle=False)

    model = ConditionalUNet(hidden_dim=args.hidden, in_channels=2)
    diffusion = Diffusion(model, n_steps=args.tsteps, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:  
            h_est, h_ideal, _ = batch
            loss = train_step(diffusion, h_ideal, h_est, optimizer)
            print(f"Epoch {epoch}, Train Loss: {loss}")
        
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

        torch.cuda.empty_cache()
        gc.collect()

    log_loss_avg = 0
    num_batch = 0

    with torch.no_grad():
    # Testing
        for batch in test_dataloader:  
            h_est, h_ideal, _ = batch
            h_est = h_est.to(diffusion.device)
            h_ideal = h_ideal.to(diffusion.device)
            generated = diffusion.sample(h_est, shape=(h_est.shape[0], 2, 120, 14))
            mse = F.mse_loss(h_ideal, generated)
            log_loss_avg += 10 * torch.log10(mse)
            print("10log Loss: ", log_loss_avg)
            num_batch += 1

    log_loss_avg /= num_batch
    print("Average 10log loss: ", log_loss_avg)
