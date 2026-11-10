import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.fft import fft2, dctn
from scipy.stats import spearmanr
import pywt

BASE_PATH = "/kaggle/input/kagglecompetition-pml"
TRAIN_PATH = f"{BASE_PATH}/train.csv"
VAL_PATH = f"{BASE_PATH}/validation.csv"
TEST_PATH = f"{BASE_PATH}/test.csv"
NPY_PATH = f"{BASE_PATH}/samples/samples"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {len(train_df)} pairs")
print(f"Val: {len(val_df)} pairs")
print(f"Test: {len(test_df)} pairs")

def load_image(img_id, npy_path):
    try:
        img_path = Path(npy_path) / f"{img_id}.npy"
        img = np.load(img_path)
        
        # Convert to float64 for numerical stability
        if img.dtype != np.float64:
            img = img.astype(np.float64)
        
        # Replace invalid values
        if not np.all(np.isfinite(img)):
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        return img
    except:
        return None


def compute_correlation(x, y):
    """
    Safely compute Pearson correlation between two arrays.
    
    Handles edge cases:
    - Unequal lengths
    - Insufficient data
    - Zero variance
    - Non-finite results
    
    Args:
        x, y: 1D numpy arrays
        
    Returns:
        float: Correlation in [-1, 1] or 0.0 if computation fails
    """
    try:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
    
        corr = np.corrcoef(x, y)[0, 1]
        return corr if np.isfinite(corr) else 0.0
    except:
        return 0.0


def extract_features(img1_id, img2_id, npy_path):
    """
    Extract correlation-based features from an image pair.
    
    Feature extraction pipeline:
    1. Load and preprocess images
    2. Compute multiple categories of features
    3. Clean and validate all features
    """
    try:
        # Load images
        img1 = load_image(img1_id, npy_path)
        img2 = load_image(img2_id, npy_path)
        
        if img1 is None or img2 is None:
            return None
        
        # Flatten images for global operations
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        h, w = img1.shape  # 256x256
        
        features = []
        
        # CATEGORY 1: GLOBAL CORRELATIONS - Measure overall similarity across entire image
        
        # Feature 1: Pearson correlation (linear relationship)
        features.append(compute_correlation(img1_flat, img2_flat))
        
        # Feature 2: Spearman correlation (rank-based, robust to outliers)
        try:
            sp, _ = spearmanr(img1_flat, img2_flat)
            features.append(sp if np.isfinite(sp) else 0.0)
        except:
            features.append(0.0)
        
        # Feature 3: Absolute correlation (magnitude similarity)
        features.append(compute_correlation(np.abs(img1_flat), np.abs(img2_flat)))
        
        # Feature 4: Squared correlation (emphasizes high-intensity regions)
        features.append(compute_correlation(img1_flat**2, img2_flat**2))
        
        # Feature 5: Sign correlation (polarity pattern matching)
        features.append(compute_correlation(np.sign(img1_flat), np.sign(img2_flat)))
        
        # Features 6-8: Quantile-based correlations (intensity-stratified)
        for q_low, q_high in [(0, 25), (25, 75), (75, 100)]:
            p_low = np.percentile(img1_flat, q_low)
            p_high = np.percentile(img1_flat, q_high)
            mask = (img1_flat >= p_low) & (img1_flat <= p_high)
            corr = compute_correlation(img1_flat[mask], img2_flat[mask]) if np.sum(mask) > 10 else 0.0
            features.append(corr)
        
        # Feature 9: Centered correlation (removes DC offset)
        features.append(compute_correlation(
            img1_flat - np.median(img1_flat), 
            img2_flat - np.median(img2_flat)
        ))
        
        # Feature 10: Min-max normalized correlation (scale-invariant)
        norm1 = (img1_flat - np.min(img1_flat)) / (np.max(img1_flat) - np.min(img1_flat) + 1e-10)
        norm2 = (img2_flat - np.min(img2_flat)) / (np.max(img2_flat) - np.min(img2_flat) + 1e-10)
        features.append(compute_correlation(norm1, norm2))
        
        # Feature 11: Histogram correlation (distribution shape similarity)
        hist1, _ = np.histogram(img1_flat, bins=50, density=True)
        hist2, _ = np.histogram(img2_flat, bins=50, density=True)
        features.append(compute_correlation(hist1, hist2))
        
        # CATEGORY 2: MULTI-SCALE PATCH CORRELATIONS - Capture patterns at different spatial resolutions
        
        # 2×2 PATCHES - Divides image into 4 quadrants (128×128 each)
        
        patch_h, patch_w = h // 2, w // 2
        
        for i in range(2):
            for j in range(2):
                # Extract patch
                p1 = img1[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].flatten()
                p2 = img2[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].flatten()
                
                # Feature: Patch correlation
                features.append(compute_correlation(p1, p2))
                
                # Feature: Mean difference
                features.append(abs(np.mean(p1) - np.mean(p2)))
                
                # Feature: Max difference
                features.append(abs(np.max(p1) - np.max(p2)))
        
        # 8×8 PATCHES - Strategic sampling of 32×32 pixel regions
        
        patch_h, patch_w = h // 8, w // 8
        
        strategic_8x8 = [
            # Top row
            (0, 0), (0, 3), (0, 7),
            # Middle rows
            (3, 0), (3, 3), (3, 7),
            (4, 0), (4, 4), (4, 7),
            # Bottom row
            (7, 0), (7, 3), (7, 7),
            # Inner corners
            (1, 1), (1, 6), (6, 1), (6, 6),
            # Inner positions
            (2, 2), (2, 5), (5, 2), (5, 5),
            # Mid edges
            (3, 1), (3, 6), (4, 1), (4, 6),
        ]
        
        for (i, j) in strategic_8x8:
            if i < 8 and j < 8:
                p1 = img1[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].flatten()
                p2 = img2[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].flatten()
                features.append(compute_correlation(p1, p2))
        
        # 4×4  PATCHES - Complete 4×4 grid (64×64 pixel patches)
        
        patch_h, patch_w = h // 4, w // 4
        
        strategic_4x4 = [
            # Border patches
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 3),
            (2, 0), (2, 3),
            (3, 0), (3, 1), (3, 2), (3, 3),
            # Center patches
            (1, 1), (1, 2), (2, 1), (2, 2),
        ]
        
        for (i, j) in strategic_4x4:
            p1 = img1[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].flatten()
            p2 = img2[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w].flatten()
            features.append(compute_correlation(p1, p2))
        
        # CATEGORY 3: SPATIAL DISTRIBUTION FEATURES - Analyze center vs periphery and diagonal patterns
        
        # CENTER VS EDGE 
        
        # Feature: Center region correlation (middle 128×128)
        center_start = h // 4
        center_end = 3 * h // 4
        center1 = img1[center_start:center_end, center_start:center_end].flatten()
        center2 = img2[center_start:center_end, center_start:center_end].flatten()
        features.append(compute_correlation(center1, center2))
        
        # Features: Four corner correlations (64×64 each)
        corner_size = h // 4
        corners = [
            (0, 0, corner_size, corner_size),                    # Top-left
            (0, w-corner_size, corner_size, w),                 # Top-right
            (h-corner_size, 0, h, corner_size),                 # Bottom-left
            (h-corner_size, w-corner_size, h, w),               # Bottom-right
        ]
        
        for (r1, c1, r2, c2) in corners:
            edge1 = img1[r1:r2, c1:c2].flatten()
            edge2 = img2[r1:r2, c1:c2].flatten()
            features.append(compute_correlation(edge1, edge2))
        
        # Feature: Center vs edge intensity comparison
        avg_edge1 = np.mean([img1[r1:r2, c1:c2].mean() for (r1, c1, r2, c2) in corners])
        avg_edge2 = np.mean([img2[r1:r2, c1:c2].mean() for (r1, c1, r2, c2) in corners])
        center_edge_diff = abs(center1.mean() - avg_edge1) - abs(center2.mean() - avg_edge2)
        features.append(center_edge_diff)
        
        # DIAGONAL PATCHES (4 features) - Patches along main diagonal (top-left to bottom-right)
        
        diag_size = h // 4
        diag_patches = [
            (0, 0, diag_size, diag_size),                       # Top-left
            (diag_size, diag_size, 2*diag_size, 2*diag_size), # Center-left
            (2*diag_size, 2*diag_size, 3*diag_size, 3*diag_size), # Center-right
            (3*diag_size, 3*diag_size, h, w),                  # Bottom-right
        ]
        
        for (r1, c1, r2, c2) in diag_patches:
            p1 = img1[r1:r2, c1:c2].flatten()
            p2 = img2[r1:r2, c1:c2].flatten()
            features.append(compute_correlation(p1, p2))
        
        
        # CATEGORY 4: FREQUENCY DOMAIN FEATURES - Analyze spectral characteristics via FFT and DCT
        
        # FFT FEATURES 
        try:
            # Compute 2D FFT
            fft1 = fft2(img1)
            fft2_img = fft2(img2)
            
            # Extract power spectra
            power1 = np.abs(fft1).flatten()
            power2 = np.abs(fft2_img).flatten()
            
            # Feature 1: Power spectrum correlation
            features.append(compute_correlation(power1, power2))
            
            # Feature 2: Log power correlation (emphasizes low frequencies)
            features.append(compute_correlation(np.log(power1 + 1), np.log(power2 + 1)))
            
            # Extract phase information
            phase1 = np.angle(fft1).flatten()
            phase2 = np.angle(fft2_img).flatten()
            
            # Feature 3: Phase correlation
            features.append(compute_correlation(phase1, phase2))
            
            # Features 4-5: Phase difference statistics
            phase_diff = phase1 - phase2
            features.append(np.mean(np.abs(phase_diff)))  # Mean difference
            features.append(np.std(phase_diff))            # Std difference
            
            # Define frequency bands (circular mask)
            center = (fft1.shape[0] // 2, fft1.shape[1] // 2)
            radius = min(center) // 2
            
            lf_mask = np.zeros(fft1.shape, dtype=bool)
            for i in range(fft1.shape[0]):
                for j in range(fft1.shape[1]):
                    if (i - center[0])**2 + (j - center[1])**2 < radius**2:
                        lf_mask[i, j] = True
            
            # Feature 6: Low-frequency correlation
            lf_power1 = power1.reshape(fft1.shape)[lf_mask].flatten()
            lf_power2 = power2.reshape(fft1.shape)[lf_mask].flatten()
            features.append(compute_correlation(lf_power1, lf_power2))
            
            # Feature 7: High-frequency correlation
            hf_power1 = power1.reshape(fft1.shape)[~lf_mask].flatten()
            hf_power2 = power2.reshape(fft1.shape)[~lf_mask].flatten()
            features.append(compute_correlation(hf_power1, hf_power2))
            
            # Feature 8: Horizontal directional correlation
            h_slice1 = power1.reshape(fft1.shape)[center[0], :]
            h_slice2 = power2.reshape(fft1.shape)[center[0], :]
            features.append(compute_correlation(h_slice1, h_slice2))
            
            # Feature 9: Vertical directional correlation
            v_slice1 = power1.reshape(fft1.shape)[:, center[1]]
            v_slice2 = power2.reshape(fft1.shape)[:, center[1]]
            features.append(compute_correlation(v_slice1, v_slice2))
            
            # Feature 10: Real component correlation
            features.append(compute_correlation(
                np.real(fft1).flatten(), 
                np.real(fft2_img).flatten()
            ))
            
            # Feature 11: Imaginary component correlation
            features.append(compute_correlation(
                np.imag(fft1).flatten(), 
                np.imag(fft2_img).flatten()
            ))
            
        except:
            features.extend([0.0] * 12)
        
        # DCT FEATURES 
        try:
            # Compute 2D DCT
            dct1 = dctn(img1, norm='ortho').flatten()
            dct2 = dctn(img2, norm='ortho').flatten()
            
            # Feature 1: Full DCT correlation
            features.append(compute_correlation(dct1, dct2))
            
            # Feature 2: Absolute DCT correlation
            features.append(compute_correlation(np.abs(dct1), np.abs(dct2)))
            
            # Separate into frequency bands
            dct1_img = dctn(img1, norm='ortho')
            dct2_img = dctn(img2, norm='ortho')
            
            # Feature 3: Low-frequency DCT correlation
            dct1_lf = dct1_img[:64, :64].flatten()
            dct2_lf = dct2_img[:64, :64].flatten()
            features.append(compute_correlation(dct1_lf, dct2_lf))
            
            # Feature 4: High-frequency DCT correlation
            dct1_hf = dct1_img[64:, 64:].flatten()
            dct2_hf = dct2_img[64:, 64:].flatten()
            corr_hf = compute_correlation(dct1_hf, dct2_hf) if len(dct1_hf) > 0 else 0.0
            features.append(corr_hf)
            
            # Feature 5: Low-frequency energy ratio
            energy_ratio = min(np.sum(dct1_lf**2), np.sum(dct2_lf**2)) / \
                          (max(np.sum(dct1_lf**2), np.sum(dct2_lf**2)) + 1e-10)
            features.append(energy_ratio)
            
            # Features 6-8: Coefficient difference statistics
            dct_diff = dct1 - dct2
            features.append(np.mean(np.abs(dct_diff)))    # Mean difference
            features.append(np.std(dct_diff))              # Std difference
            features.append(np.median(np.abs(dct_diff)))  # Median difference
            
        except:
            features.extend([0.0] * 8)
        
        # CATEGORY 5: WAVELET FEATURES (10 features) - Multi-resolution analysis via Haar wavelets
        try:
            # Single-level 2D wavelet decomposition
            coeffs1 = pywt.dwt2(img1, 'haar')
            coeffs2 = pywt.dwt2(img2, 'haar')
            
            # Extract coefficient matrices
            cA1, (cH1, cV1, cD1) = coeffs1  
            cA2, (cH2, cV2, cD2) = coeffs2
            
            # Features 1-4: Coefficient correlations
            features.append(compute_correlation(cA1.flatten(), cA2.flatten()))  # Approximation
            features.append(compute_correlation(cH1.flatten(), cH2.flatten()))  # Horizontal
            features.append(compute_correlation(cV1.flatten(), cV2.flatten()))  # Vertical
            features.append(compute_correlation(cD1.flatten(), cD2.flatten()))  # Diagonal
            
            # Feature 5: Approximation energy ratio
            energy_A1 = np.sum(cA1**2)
            energy_A2 = np.sum(cA2**2)
            features.append(min(energy_A1, energy_A2) / (max(energy_A1, energy_A2) + 1e-10))
            
            # Feature 6: Horizontal detail energy ratio
            energy_H1 = np.sum(cH1**2)
            energy_H2 = np.sum(cH2**2)
            features.append(min(energy_H1, energy_H2) / (max(energy_H1, energy_H2) + 1e-10))
            
            # Features 7-10: Detail coefficient statistics
            features.append(abs(np.mean(cH1) - np.mean(cH2)))  # Horizontal mean diff
            features.append(abs(np.mean(cV1) - np.mean(cV2)))  # Vertical mean diff
            features.append(abs(np.mean(cD1) - np.mean(cD2)))  # Diagonal mean diff
            features.append(abs(np.std(cH1) - np.std(cH2)))    # Horizontal std diff
            
        except:
            features.extend([0.0] * 10)
        
        # CATEGORY 6: STATISTICAL MOMENTS - Basic distribution property differences
        
        # Features 1-3: Central tendency
        features.append(abs(np.mean(img1_flat) - np.mean(img2_flat)))      # Mean diff
        features.append(abs(np.std(img1_flat) - np.std(img2_flat)))        # Std diff
        features.append(abs(np.median(img1_flat) - np.median(img2_flat)))  # Median diff
        
        # Features 4-5: Shape descriptors
        skew1, skew2 = stats.skew(img1_flat), stats.skew(img2_flat)
        kurt1, kurt2 = stats.kurtosis(img1_flat), stats.kurtosis(img2_flat)
        
        skew_diff = abs(skew1 - skew2) if np.isfinite(skew1) and np.isfinite(skew2) else 0.0
        kurt_diff = abs(kurt1 - kurt2) if np.isfinite(kurt1) and np.isfinite(kurt2) else 0.0
        features.append(skew_diff)
        features.append(kurt_diff)
        
        # Features 6-8: Percentile differences
        for p in [25, 50, 75]:
            p_diff = abs(np.percentile(img1_flat, p) - np.percentile(img2_flat, p))
            features.append(p_diff)
        
        # Features 9-10: Extreme values
        features.append(abs(np.min(img1_flat) - np.min(img2_flat)))  # Min diff
        features.append(abs(np.max(img1_flat) - np.max(img2_flat)))  # Max diff
        
        # CATEGORY 7: RANK CORRELATIONS - Ordinal relationship preservation
        
        # Compute ranks
        rank1 = stats.rankdata(img1_flat)
        rank2 = stats.rankdata(img2_flat)
        
        # Feature 1: Global rank correlation
        features.append(compute_correlation(rank1, rank2))
        
        # Features 2-3: Rank difference statistics
        rank_diff = rank1 - rank2
        features.append(np.mean(np.abs(rank_diff)))  # Mean rank difference
        features.append(np.std(rank_diff))            # Std rank difference
        
        # Feature 4: Center region rank correlation
        center_rank1 = stats.rankdata(center1)
        center_rank2 = stats.rankdata(center2)
        features.append(compute_correlation(center_rank1, center_rank2))
        
        # Feature 5: Percentile rank correlation
        percentiles = [10, 25, 50, 75, 90]
        p1_vals = [np.percentile(img1_flat, p) for p in percentiles]
        p2_vals = [np.percentile(img2_flat, p) for p in percentiles]
        features.append(compute_correlation(p1_vals, p2_vals))
        
        # Feature 6: High-value region rank correlation
        high_mask = img1_flat > np.percentile(img1_flat, 75)
        if np.sum(high_mask) > 10:
            features.append(compute_correlation(rank1[high_mask], rank2[high_mask]))
        else:
            features.append(0.0)
        
        # CATEGORY 8: OVERLAP PATCHES - Sliding window analysis with 50% overlap
        
        stride = h // 8        # 32 pixels
        window_size = h // 4   # 64×64 windows
        
        overlap_positions = [
            (0, 0),                                            # Top-left
            (stride, stride),                                  # Offset top-left
            (h//2 - window_size//2, w//2 - window_size//2),  # Center
            (stride*3, stride*3),                             # Offset center
            (h - window_size, 0),                             # Bottom-left
            (0, w - window_size),                             # Top-right
            (h - window_size, w - window_size),               # Bottom-right
        ]
        
        for (r, c) in overlap_positions:
            if r + window_size <= h and c + window_size <= w:
                p1 = img1[r:r+window_size, c:c+window_size].flatten()
                p2 = img2[r:r+window_size, c:c+window_size].flatten()
                features.append(compute_correlation(p1, p2))
            else:
                features.append(0.0)
        
        # Convert to array and clean
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) 
        
        return features.tolist()
        
    except Exception as e:
        print(f"Error processing {img1_id}, {img2_id}: {e}")
        return None


def extract_features_from_df(df, npy_path, name="dataset"):
    """
    Extract features for all pairs in a dataframe.
        
    Returns:
        tuple: (X, y, ids) where X is feature matrix, y is labels, ids is pair identifiers
    """
    features_list = []
    labels = []
    ids = []
    
    total = len(df)
    print(f"\nProcessing {name} ({total} pairs)...")
    
    
    for idx, row in df.iterrows():
        # Progress logging
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{total}")
        
        # Extract features
        feats = extract_features(row['id_noise_1'], row['id_noise_2'], npy_path)
        
        if feats is not None:
            features_list.append(feats)
            if 'label' in row:
                labels.append(row['label'])
            ids.append((row['id_noise_1'], row['id_noise_2']))
    
    # Convert to arrays
    X = np.array(features_list)
    y = np.array(labels) if labels else None
    
    print(f"{name}: {X.shape[0]} samples, {X.shape[1]} features ")
    
    return X, y, ids

# Extract features for all datasets
X_train, y_train, train_ids = extract_features_from_df(train_df, NPY_PATH, "TRAIN")
X_val, y_val, val_ids = extract_features_from_df(val_df, NPY_PATH, "VALIDATION")
X_test, _, test_ids = extract_features_from_df(test_df, NPY_PATH, "TEST")

np.save('X_train_v6_final.npy', X_train)
np.save('y_train_v6_final.npy', y_train)
np.save('X_val_v6_final.npy', X_val)
np.save('y_val_v6_final.npy', y_val)
np.save('X_test_v6_final.npy', X_test)
np.save('test_ids_v6_final.npy', test_ids)

print("EXTRACTION COMPLETE")
print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")
print(f"Test: {X_test.shape}")