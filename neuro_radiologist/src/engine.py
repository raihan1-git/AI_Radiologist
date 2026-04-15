import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    
    # Initialize the AMP context for bfloat16 (Optimized for A100)
    # Note: If testing locally on a CPU or older GPU, this will safely fallback or use standard float16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Progress bar for the console
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
    
    for batch in pbar:
        # 1. Move data to GPU/CPU
        images = batch["image"].to(device)
        
        # 2. Zero the gradients
        optimizer.zero_grad()
        
        # 3. Forward pass with AMP
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32):
            loss, pred, mask = model(images, mask_ratio=0.75)
            
        # 4. Backward pass (scaling the loss to prevent underflow)
        scaler.scale(loss).backward()
        
        # 5. Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # 6. Logging
        loss_value = loss.item()
        total_loss += loss_value
        pbar.set_postfix({"MSE Loss": f"{loss_value:.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss