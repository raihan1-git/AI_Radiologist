import torch
import torch.optim as optim
from neuro_radiologist.src.data_pipeline import get_mri_dataloader
from neuro_radiologist.src.models.mae_3d import MaskedAutoencoder3D
from neuro_radiologist.src.engine import train_one_epoch

# 1. Setup Data (using our dummy file repeated to make a small dataset)
file_paths = ['neuro_radiologist/data/raw/dummy_mri.nii.gz'] * 8 
dataloader = get_mri_dataloader(file_paths, batch_size=2, num_workers=0)

# 2. Setup Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = MaskedAutoencoder3D().to(device)

# 3. Setup Optimizer (AdamW is standard for Transformers)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# 4. Run a tiny training loop
print("\n--- Starting Mock Training ---")
for epoch in range(1, 6): # Run for 5 epochs
    avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
    print(f"Epoch {epoch} Completed | Average Loss: {avg_loss:.4f}\n")