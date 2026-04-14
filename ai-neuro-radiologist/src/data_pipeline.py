import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Orientationd, Spacingd, ScaleIntensityd
)

def get_mri_dataloader(file_paths, batch_size=2, num_workers=2):
    """
    Creates an optimized MONAI DataLoader with caching.
    """
    # 1. Format the file paths into a list of dictionaries
    data_dicts = [{"image": path} for path in file_paths]
    
    # 2. Define the pipeline
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        ScaleIntensityd(keys=["image"])
    ])
    
    # 3. Create the CacheDataset
    # It will preprocess all files in data_dicts up to the final transform
    print("Caching dataset into RAM...")
    dataset = CacheDataset(
        data=data_dicts, 
        transform=transforms, 
        cache_rate=1.0, # Cache 100% of the data
        num_workers=num_workers
    )
    
    # 4. Wrap in a PyTorch DataLoader
    # This handles batching the 3D volumes together
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    return loader