"""
Hyperparameter Tuning for LightGBM
Testing different configurations to find the best settings
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr, kendalltau
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

X_train = np.load('final_features/X_train_v6_final.npy')
y_train = np.load('final_features/y_train_v6_final.npy')
X_val = np.load('final_features/X_val_v6_final.npy')
y_val = np.load('final_features/y_val_v6_final.npy')

print(f"Loaded {X_train.shape[0]} training samples")
print(f"Loaded {X_val.shape[0]} validation samples\n")

# clean any weird values
X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)

# standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# configurations to test - varying one parameter at a time
configs = [
    # test different number of trees
    {"name": "n_est_800", "n_estimators": 800, "learning_rate": 0.015, "num_leaves": 255, "max_depth": 15},
    {"name": "n_est_1200", "n_estimators": 1200, "learning_rate": 0.015, "num_leaves": 255, "max_depth": 15},
    {"name": "n_est_1500", "n_estimators": 1500, "learning_rate": 0.015, "num_leaves": 255, "max_depth": 15},
    
    # test different learning rates
    {"name": "lr_0.01", "n_estimators": 1200, "learning_rate": 0.01, "num_leaves": 255, "max_depth": 15},
    {"name": "lr_0.015", "n_estimators": 1200, "learning_rate": 0.015, "num_leaves": 255, "max_depth": 15},
    {"name": "lr_0.02", "n_estimators": 1200, "learning_rate": 0.02, "num_leaves": 255, "max_depth": 15},
    
    # test different tree sizes
    {"name": "leaves_127", "n_estimators": 1200, "learning_rate": 0.015, "num_leaves": 127, "max_depth": 12},
    {"name": "leaves_255", "n_estimators": 1200, "learning_rate": 0.015, "num_leaves": 255, "max_depth": 15},
    {"name": "leaves_511", "n_estimators": 1200, "learning_rate": 0.015, "num_leaves": 511, "max_depth": 18},
    
    # test different regularization strengths
    {"name": "reg_weak", "n_estimators": 1200, "learning_rate": 0.015, "num_leaves": 255, "max_depth": 15,
     "reg_alpha": 0.0, "reg_lambda": 0.0},
    {"name": "reg_medium", "n_estimators": 1200, "learning_rate": 0.015, "num_leaves": 255, "max_depth": 15,
     "reg_alpha": 0.2, "reg_lambda": 0.2},
    {"name": "reg_strong", "n_estimators": 1200, "learning_rate": 0.015, "num_leaves": 255, "max_depth": 15,
     "reg_alpha": 0.5, "reg_lambda": 0.5},
]

print(f"Testing {len(configs)} different configurations...\n")

results = []

for cfg in configs:
    name = cfg.pop("name")
    print(f"Testing {name}...")
    
    # create model with this config
    model = lgb.LGBMClassifier(
        **cfg,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=7,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # train and get predictions
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_val_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    # calculate metrics
    acc = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)
    spear, _ = spearmanr(probs, y_val)
    kend, _ = kendalltau(probs, y_val)
    
    print(f"  Accuracy: {acc*100:.2f}% | Spearman: {spear:.4f}\n")
    
    results.append({
        "config": name,
        "n_estimators": cfg.get("n_estimators"),
        "learning_rate": cfg.get("learning_rate"),
        "num_leaves": cfg.get("num_leaves"),
        "max_depth": cfg.get("max_depth"),
        "reg_alpha": cfg.get("reg_alpha", 0.2),  
        "reg_lambda": cfg.get("reg_lambda", 0.2),
        "accuracy": acc,
        "auc": auc,
        "spearman": spear,
        "kendall": kend
    })

df = pd.DataFrame(results)
df.to_csv("hyperparameter_results.csv", index=False)

print("RESULTS BY PARAMETER")

# show effect of number of estimators
print("\nNumber of Estimators:")
n_est = df[df["config"].str.contains("n_est")]
for _, row in n_est.iterrows():
    print(f"  {row['n_estimators']} trees: {row['accuracy']*100:.2f}%")

# show effect of learning rate
print("\nLearning Rate:")
lr = df[df["config"].str.contains("lr_")]
for _, row in lr.iterrows():
    print(f"  lr={row['learning_rate']}: {row['accuracy']*100:.2f}%")

# show effect of tree complexity
print("\nTree Complexity (num leaves):")
leaves = df[df["config"].str.contains("leaves")]
for _, row in leaves.iterrows():
    print(f"  {row['num_leaves']} leaves: {row['accuracy']*100:.2f}%")

# show effect of regularization
print("\nRegularization:")
reg = df[df["config"].str.contains("reg_")]
for _, row in reg.iterrows():
    strength = "none" if row['reg_alpha'] == 0 else f"alpha={row['reg_alpha']}"
    print(f"  {strength}: {row['accuracy']*100:.2f}%")

