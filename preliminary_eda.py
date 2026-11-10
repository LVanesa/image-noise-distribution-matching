# %%
sample_submission_path = "image-noise-distribution-matching/Archive/sample_submission.csv"
test_path = "image-noise-distribution-matching/Archive/test.csv"
train_path = "image-noise-distribution-matching/Archive/train.csv"
validation_path = "image-noise-distribution-matching/Archive/validation.csv"
npy_images_path = "image-noise-distribution-matching/Archive/samples"

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy import stats
from scipy.fft import fft2, fftshift
import warnings
warnings.filterwarnings('ignore')


# %%
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)

# %%
print(f"\nDATASETS DIMENSIONS:")
print(f"   Training:   {len(train_df):,} samples")
print(f"   Validation: {len(val_df):,} samples")
print(f"   Test:       {len(test_df):,} samples")

# %%
train_df.head(10)


# %%
def show_label_stats(df, dataset_name):
    print(f"\n{dataset_name} Label Distribution ")
    
    counts = df['label'].value_counts()
    total = len(df)
    
    for label, count in counts.items():
        percentage = (count / total) * 100
        label_description = "Different Distribution" if label == 0 else "Same Distribution"
        print(f"Label {label} ({label_description}): {count:,} ({percentage:.2f}%)")

show_label_stats(train_df, "TRAIN")
show_label_stats(val_df, "VALIDATION")

# %%
npy_files = list(Path(npy_images_path).glob("*.npy"))
print(f"\n Total .npy images: {len(npy_files):,}")

# %%
# Loading a few samples to check the format
samples = []
sample_ids = []

for npy_path in npy_files[:5]:
    img = np.load(npy_path)
    samples.append(img)
    sample_ids.append(npy_path.stem)

# Check first sample to verify dimensions and scaling
first_img = samples[0]

print("\n--- Noise Image Properties ---")
print(f"Shape:         {first_img.shape}")
print(f"Data type:     {first_img.dtype}")
print(f"Min value:     {first_img.min():.4f}")
print(f"Max value:     {first_img.max():.4f}")
print(f"Mean:          {first_img.mean():.4f}")
print(f"Std Deviation: {first_img.std():.4f}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sample Noise Images', fontsize=16, fontweight='bold')

flat_axes = axes.flatten()

for i in range(len(samples)):
    if i >= 6: break 
    ax = flat_axes[i]
    img = samples[i]
    im = ax.imshow(img, cmap='gray')
    ax.set_title(f"ID: {sample_ids[i][:8]}...\nShape: {img.shape}", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for j in range(len(samples), len(flat_axes)):
    flat_axes[j].axis('off')

plt.tight_layout()
plt.savefig('noise_samples.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
def load_image_by_id(img_id):
    img_path = Path(npy_images_path) / f"{img_id}.npy"
    if img_path.exists():
        return np.load(img_path)
    return None

# %%
# Find a pair with label=1 (matching) and another one with label=0 (non-matching)
matching_pair = train_df[train_df['label'] == 1].iloc[0]
non_matching_pair = train_df[train_df['label'] == 0].iloc[0]

print(f"\n PAIR MATCHING (label=1):")
print(f"   ID1: {matching_pair['id_noise_1']}")
print(f"   ID2: {matching_pair['id_noise_2']}")

print(f"\n PAIR NON-MATCHING (label=0):")
print(f"   ID1: {non_matching_pair['id_noise_1']}")
print(f"   ID2: {non_matching_pair['id_noise_2']}")


# %%
match_img1 = load_image_by_id(matching_pair['id_noise_1'])
match_img2 = load_image_by_id(matching_pair['id_noise_2'])
nomatch_img1 = load_image_by_id(non_matching_pair['id_noise_1'])
nomatch_img2 = load_image_by_id(non_matching_pair['id_noise_2'])

# %%
# Comparative Visualization: Label 1 (Match) vs Label 0 (No Match)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Pair Comparison: Matching (1) vs Non-Matching (0)', fontsize=16, fontweight='bold')

pairs = [
    (match_img1, match_img2, "MATCHING", 0),      # Row 0
    (nomatch_img1, nomatch_img2, "NON-MATCHING", 1) # Row 1
]

for img1, img2, label_text, row in pairs:
    if img1 is not None and img2 is not None:
        # Show raw images
        axes[row, 0].imshow(img1, cmap='gray')
        axes[row, 0].set_title(f'{label_text} - Img 1')
        
        axes[row, 1].imshow(img2, cmap='gray')
        axes[row, 1].set_title(f'{label_text} - Img 2')
        
        # Calculate and show absolute difference
        diff = np.abs(img1 - img2)
        im = axes[row, 2].imshow(diff, cmap='hot')
        axes[row, 2].set_title(f'Abs Diff (Mean: {diff.mean():.4f})')
        plt.colorbar(im, ax=axes[row, 2])
        
        # Histogram Overlay
        axes[row, 3].hist(img1.flatten(), bins=50, alpha=0.5, label='Img 1', color='blue')
        axes[row, 3].hist(img2.flatten(), bins=50, alpha=0.5, label='Img 2', color='red')
        axes[row, 3].set_title('Histogram Overlay')
        axes[row, 3].legend()
        axes[row, 3].set_xlabel('Pixel Value')

        for col in range(3):
            axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('pair_comparison.png', dpi=150)
plt.show()

# %% [markdown]
# # Statistical Features 

# %%
def extract_statistical_features(image):
    """
    Extract statistical features from a noise image.
    This function computes statistical metrics that characterize the
    distribution and structure of pixel values in the image. Features include
    central tendency, dispersion, shape, percentiles, and energy metrics.
    """
    
    # convert to float32 to prevent overflow during computation
    img_float = image.astype(np.float32)
    
    pixels = img_float.flatten()
    pixels_valid = pixels[np.isfinite(pixels)]
    
    if len(pixels_valid) == 0:
        feature_names = [
            'mean', 'std', 'variance', 'median', 'min', 'max', 'range',
            'skewness', 'kurtosis', 'q10', 'q25', 'q75', 'q90', 'iqr',
            'energy', 'rms'
        ]
        return {name: 0.0 for name in feature_names}
    
    try:
        features = {
            # Central tendency measures
            'mean': float(np.mean(pixels_valid)),
            'median': float(np.median(pixels_valid)),
            
            # Dispersion measures
            'std': float(np.std(pixels_valid)),
            'variance': float(np.var(pixels_valid)),
            'range': float(np.ptp(pixels_valid)),  #peak-to-peak=max-min
            
            # Extreme values
            'min': float(np.min(pixels_valid)),
            'max': float(np.max(pixels_valid)),
            
            # Percentile-based measures
            'q10': float(np.percentile(pixels_valid, 10)),
            'q25': float(np.percentile(pixels_valid, 25)),
            'q75': float(np.percentile(pixels_valid, 75)),
            'q90': float(np.percentile(pixels_valid, 90)),
            'iqr': float(
                np.percentile(pixels_valid, 75) - np.percentile(pixels_valid, 25)
            ),
            
            # Higher-order moments (shape of distribution)
            'skewness': float(pd.Series(pixels_valid).skew()),
            'kurtosis': float(pd.Series(pixels_valid).kurtosis()),
            
            # Energy metrics
            'energy': float(np.mean(pixels_valid ** 2)),  # Mean square energy
            'rms': float(np.sqrt(np.mean(pixels_valid ** 2))),  # Root mean square
        }
        
        features = {
            name: (value if np.isfinite(value) else 0.0)
            for name, value in features.items()
        }
        
    except Exception as e:
        print(f"Warning: Feature extraction failed - {str(e)}")
        feature_names = [
            'mean', 'std', 'variance', 'median', 'min', 'max', 'range',
            'skewness', 'kurtosis', 'q10', 'q25', 'q75', 'q90', 'iqr',
            'energy', 'rms'
        ]
        features = {name: 0.0 for name in feature_names}
    
    return features

# %%
if match_img1 is not None:
    print("\nSTATS FOR MATCHING PAIR:")
    stats1 = extract_statistical_features(match_img1)
    stats2 = extract_statistical_features(match_img2)
    
    print(f"\n   Image 1:")
    for key, val in stats1.items():
        print(f"      {key:12s}: {val:.6f}")
    
    print(f"\n   Image 2:")
    for key, val in stats2.items():
        print(f"      {key:12s}: {val:.6f}")
    
    print(f"\n   Abs Difference:")
    for key in stats1.keys():
        diff = abs(stats1[key] - stats2[key])
        print(f"      Δ{key:11s}: {diff:.6f}")


# %%

if nomatch_img1 is not None:
    print("\n STATS FOR NON-MATCHING PAIR:")
    stats1 = extract_statistical_features(nomatch_img1)
    stats2 = extract_statistical_features(nomatch_img2)
    
    print(f"\n   Image 1:")
    for key, val in stats1.items():
        print(f"      {key:12s}: {val:.6f}")
    
    print(f"\n   Image 2:")
    for key, val in stats2.items():
        print(f"      {key:12s}: {val:.6f}")
    
    print(f"\n   Abs Difference:")
    for key in stats1.keys():
        diff = abs(stats1[key] - stats2[key])
        print(f"      Δ{key:11s}: {diff:.6f}")

# %%
if match_img1 is not None and match_img2 is not None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Frequency Domain Analysis (FFT)', fontsize=16, fontweight='bold')
    
    # FFT pentru matching pair
    fft1 = fftshift(fft2(match_img1))
    fft2_img = fftshift(fft2(match_img2))
    
    magnitude1 = np.log(np.abs(fft1) + 1)
    magnitude2 = np.log(np.abs(fft2_img) + 1)
    
    im1 = axes[0, 0].imshow(magnitude1, cmap='viridis')
    axes[0, 0].set_title('MATCHING - Image 1 (FFT)', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(magnitude2, cmap='viridis')
    axes[0, 1].set_title('MATCHING - Image 2 (FFT)', fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # FFT pentru non-matching pair
    if nomatch_img1 is not None and nomatch_img2 is not None:
        fft3 = fftshift(fft2(nomatch_img1))
        fft4 = fftshift(fft2(nomatch_img2))
        
        magnitude3 = np.log(np.abs(fft3) + 1)
        magnitude4 = np.log(np.abs(fft4) + 1)
        
        im3 = axes[1, 0].imshow(magnitude3, cmap='viridis')
        axes[1, 0].set_title('NON-MATCHING - Image 1 (FFT)', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(magnitude4, cmap='viridis')
        axes[1, 1].set_title('NON-MATCHING - Image 2 (FFT)', fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('fft_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


