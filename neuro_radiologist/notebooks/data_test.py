import numpy as np
import nibabel as nib
import os

# 1. Create a dummy 3D volume (e.g., 64x64x64)
# Let's make a sphere in the center to simulate a "brain" or "tumor"
shape = (64, 64, 64)
data = np.zeros(shape, dtype=np.float32)

# Generate coordinate grids
x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
center = (32, 32, 32)
radius = 20

# Create a sphere mask
mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
data[mask] = 255.0 # Give the "brain" some intensity

# 2. Define a basic Identity Affine matrix
# An identity matrix means 1 voxel = 1mm in all directions
affine = np.eye(4)

# 3. Save as a NIfTI file
os.makedirs('neuro_radiologist/data/raw', exist_ok=True)
file_path = 'neuro_radiologist/data/raw/dummy_mri.nii.gz'
nifti_img = nib.Nifti1Image(data, affine)
nib.save(nifti_img, file_path)
print(f"Saved dummy NIfTI to: {file_path}")

# 4. Load it back and inspect
loaded_img = nib.load(file_path)
loaded_data = loaded_img.get_fdata()
loaded_affine = loaded_img.affine

print("\n--- NIfTI Inspection ---")
print(f"Volume Shape: {loaded_data.shape}")
print(f"Data Type: {loaded_data.dtype}")
print(f"Affine Matrix:\n{loaded_affine}")


from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd
)

# 1. Define the dictionary we will pass to the pipeline
# In real training, we would also have a "label" key here.
data_dict = {"image": file_path}

# 2. Build the MONAI Transform Pipeline
preprocessing_pipeline = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    # RAS: Right, Anterior, Superior (Standard anatomical orientation)
    Orientationd(keys=["image"], axcodes="RAS"),
    # Resample all voxels to be exactly 1mm x 1mm x 1mm
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    # Normalize intensities to [0, 1]
    ScaleIntensityd(keys=["image"])
])

# 3. Pass the data dictionary through the pipeline
processed_data = preprocessing_pipeline(data_dict)

# 4. Extract the processed image tensor
final_tensor = processed_data["image"]

print("\n--- MONAI Pipeline Output ---")
print(f"Final Tensor Type: {type(final_tensor)}")
print(f"Final Tensor Shape: {final_tensor.shape}")
print(f"Intensity Min/Max: {final_tensor.min():.2f} / {final_tensor.max():.2f}")


from neuro_radiologist.src.data_pipeline import get_mri_dataloader

# Simulate having 4 patient scans by repeating the dummy file
file_paths = ['neuro_radiologist/data/raw/dummy_mri.nii.gz'] * 4 

# Get the dataloader (batch size 2)
dataloader = get_mri_dataloader(file_paths, batch_size=2, num_workers=0)

# Fetch one batch
for batch in dataloader:
    images = batch["image"]
    print(f"\nBatch Tensor Shape: {images.shape}")
    break # We only need to check the first batch