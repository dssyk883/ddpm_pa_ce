import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import MatDataset
from torch.utils.data import DataLoader
import argparse
import gc
from ddpm import ConditionalUnet, Diffusion

def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model weights')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tsteps', type=int, default=500)
    parser.add_argument('--hidden', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def calculate_nmse(generated, h_ideal, h_est):
    error = generated - h_ideal 
    error_power = torch.sum(torch.mul(error, error))  
    pilot_power = torch.sum(torch.mul(h_est, h_est))
    
    nmse = error_power / pilot_power
    
    return nmse

def test_model():
    args = parse_params()
    
    # Load test dataset
    mat_testset = MatDataset(
        data_dir=args.test_dir,
        pilot_dims=(18, 2),
        transform=None,
        return_type="2channel"
    )

    test_dataloader = DataLoader(mat_testset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model and load weights
    model = ConditionalUnet(hidden_dim=args.hidden, in_channels=2)
    model.load_state_dict(torch.load(args.model_path))
    diffusion = Diffusion(model, n_steps=args.tsteps, device=args.device)
    
    # Testing loop
    model.eval()
    total_nmse = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            h_est, h_ideal, _ = batch
            h_est = h_est.to(args.device)
            h_ideal = h_ideal.to(args.device)
            
            # Generate samples
            generated = diffusion.sample(h_est, shape=(h_est.shape[0], 2, 120, 14))
            
            # Calculate NMSE
            nmse = calculate_nmse(generated, h_ideal, h_est)
            log_nmse = 10 * torch.log10(nmse)
            
            total_nmse += log_nmse.item()
            num_batches += 1
            
            print(f"Batch {num_batches}, 10log10(NMSE): {log_nmse.item():.4f} dB")
            
            # Free up memory
            torch.cuda.empty_cache()
            gc.collect()
    
    average_nmse = total_nmse / num_batches
    print(f"\nFinal Test Results:")
    print(f"Average 10log10(NMSE): {average_nmse:.4f} dB")
    
    # Save results to file
    with open('ddpm_test_results.txt', 'w') as f:
        f.write(f"Test Configuration:\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Test directory: {args.test_dir}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Time steps: {args.tsteps}\n")
        f.write(f"Hidden dimension: {args.hidden}\n\n")
        f.write(f"Average 10log10(NMSE): {average_nmse:.4f} dB\n")

if __name__ == "__main__":
    test_model()