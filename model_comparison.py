import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import spearmanr, kendalltau
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

X_train = np.load('final_features/X_train_v6_final.npy')
y_train = np.load('final_features/y_train_v6_final.npy')
X_val = np.load('final_features/X_val_v6_final.npy')
y_val = np.load('final_features/y_val_v6_final.npy')

print(f"Loaded training set: {X_train.shape}")
print(f"Loaded validation set: {X_val.shape}\n")

# clean any invalid values (just in case)
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# different models configurations to compare
models = {
    "LightGBM_baseline": lgb.LGBMClassifier(
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
    ),
    
    "LightGBM_deep": lgb.LGBMClassifier(
        n_estimators=1500,
        num_leaves=300,
        learning_rate=0.012,
        max_depth=18,
        min_child_samples=8,
        feature_fraction=0.75,
        bagging_fraction=0.75,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    
    "XGBoost_baseline": xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=12,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    ),
    
    "XGBoost_deep": xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.012,
        max_depth=15,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    ),
    
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ),
    
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=10,
        random_state=42
    )
}


results = []

# train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    
    model.fit(X_train_scaled, y_train)
    
    # get predictions on validation set
    val_prob = model.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_prob > 0.5).astype(int)
    
    # calculate classification metrics
    acc = accuracy_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_prob)
    f1 = f1_score(y_val, val_pred)
    
    # correlation metrics
    spearman, _ = spearmanr(val_prob, y_val)
    kendall, _ = kendalltau(val_prob, y_val)
    
    print(f"  Accuracy: {acc*100:.2f}% | AUC: {auc:.4f} | Spearman: {spearman:.4f}\n")
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "AUC": auc,
        "F1": f1,
        "Spearman": spearman,
        "Kendall": kendall
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Accuracy", ascending=False)

results_df.to_csv("validation_results.csv", index=False)
print("Validation Results:")
print(results_df)


