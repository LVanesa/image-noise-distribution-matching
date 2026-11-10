"""
Final Training and Submission Generation
Trains LightGBM and XGBoost on full dataset and creates submission files
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# load all the data
X_train = np.load('final_features\X_train_v6_final.npy')
y_train = np.load('final_features\y_train_v6_final.npy')
X_val = np.load('final_features\X_val_v6_final.npy')
y_val = np.load('final_features\y_val_v6_final.npy')
X_test = np.load('final_features\X_test_v6_final.npy')
test_ids = np.load('final_features\test_ids_v6_final.npy', allow_pickle=True)


print(f"Train: {X_train.shape[0]} samples")
print(f"Val: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples\n")

# clean up any nans
X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)
X_test = np.nan_to_num(X_test)

# combine train and validation for final training
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])
print(f"Combined dataset: {X_full.shape[0]} samples\n")

# standardize everything
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
X_test_scaled = scaler.transform(X_test)

# train lightgbm (had best validation accuracy)
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=1200,
    num_leaves=255,
    learning_rate=0.015,
    max_depth=15,
    min_child_samples=10,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    bagging_freq=7,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(X_full_scaled, y_full)
lgb_probs = lgb_model.predict_proba(X_test_scaled)[:, 1]

# train xgboost (had best auc)
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.012,
    max_depth=15,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(X_full_scaled, y_full)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
print("  Done!\n")

# create ensemble (60% lightgbm, 40% xgboost)
ensemble_probs = 0.6 * lgb_probs + 0.4 * xgb_probs

# generate submissions at different thresholds
print("Generating submission files...")

def create_submissions(probs, base_name):
    thresholds = [0.48, 0.50, 0.52]
    
    for t in thresholds:
        preds = (probs > t).astype(int)
        
        submission = pd.DataFrame({
            'id_pair': [f'({a},{b})' for a, b in test_ids],
            'label': preds
        })
        
        filename = f"{base_name}_t{int(t*100)}.csv"
        submission.to_csv(filename, index=False)
        
        n_positive = preds.sum()
        pct_positive = n_positive / len(preds) * 100
        print(f"  {filename}: {n_positive} positives ({pct_positive:.1f}%)")

# generate all submissions
create_submissions(lgb_probs, "submission_V6_LGB")
print()
create_submissions(xgb_probs, "submission_V6_XGB")
print()
create_submissions(ensemble_probs, "submission_V6_ENSEMBLE")

print("All submissions generated successfully!")
