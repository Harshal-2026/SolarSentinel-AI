import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_squared_error, r2_score

CLASS_NAMES = ['B-class', 'C-class', 'M-class', 'X-class']

def calculate_lead_time(y_true, y_proba, threshold=0.5):
    """
    Calculates the average lead time in minutes.
    Assumes temporal order (1 minute per index).
    A flare event is a contiguous block where y_true >= 1.
    For each event block, find the first index where P(flare) >= threshold.
    Lead time is the distance from the alert index to the peak of the event block.
    """
    p_flare = y_proba[:, 1:].sum(axis=1)
    
    in_flare = False
    event_start = 0
    events = []
    
    for i, label in enumerate(y_true):
        if label >= 1 and not in_flare:
            in_flare = True
            event_start = i
        elif label == 0 and in_flare:
            in_flare = False
            events.append((event_start, i-1))
            
    if in_flare:
        events.append((event_start, len(y_true)-1))
        
    lead_times = []
    for start, end in events:
        # y_true is 'max in next 60 min'. So the peak is approx at start + 60.
        peak_idx = start + 60
        search_start = max(0, start - 120) # search up to 2 hours before
        search_end = min(len(y_true), peak_idx)
        
        alert_idx = -1
        for i in range(search_start, search_end):
            if p_flare[i] >= threshold:
                alert_idx = i
                break
                
        if alert_idx != -1 and alert_idx < peak_idx:
            lead_times.append(peak_idx - alert_idx)
            
    if lead_times:
        return np.mean(lead_times), lead_times
    return 0.0, []

def main():
    base_dir = os.path.dirname(__file__)
    x_path = os.path.join(base_dir, "X_data.npy")
    y_path = os.path.join(base_dir, "y_labels.npy")
    y_impact_path = os.path.join(base_dir, "y_impact.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path) or not os.path.exists(y_impact_path):
        print("Data files not found. Please run create_ml_dataset.py first.")
        return

    print("Loading dual-channel dataset with spatial features...")
    X = np.load(x_path)
    y = np.load(y_path)
    y_impact = np.load(y_impact_path)

    samples = X.shape[0]
    X_2d = X.reshape(samples, -1)
    print(f"Flattened X shape for XGBoost: {X_2d.shape}")

    # Split WITHOUT shuffling to preserve temporal order for Lead Time calculation
    print("Splitting dataset (80/20) temporally...")
    split_idx = int(samples * 0.8)
    X_train, X_test = X_2d[:split_idx], X_2d[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    y_impact_train, y_impact_test = y_impact[:split_idx], y_impact[split_idx:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    print("\nTraining XGBoost Multi-Class model (softprob) for Flare Class...")
    model_class = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
    )
    model_class.fit(X_train, y_train)

    print("\nTraining XGBoost Regressor for Earth Impact Probability...")
    model_impact = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42,
    )
    model_impact.fit(X_train, y_impact_train)

    y_pred = model_class.predict(X_test)
    y_proba = model_class.predict_proba(X_test)
    y_impact_pred = model_impact.predict(X_test)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n========== Flare Class Model Evaluation (Macro Avg) ==========")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Impact Metrics
    rmse = np.sqrt(mean_squared_error(y_impact_test, y_impact_pred))
    r2 = r2_score(y_impact_test, y_impact_pred)
    print("\n========== Impact Probability Model Evaluation ==========")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    
    # Lead Time
    avg_lead_time, all_lead_times = calculate_lead_time(y_test, y_proba, threshold=0.5)
    print(f"\n========== Forecasting Lead Time ==========")
    print(f"  Average Lead Time: {avg_lead_time:.1f} minutes")
    print(f"  Events analyzed:   {len(all_lead_times)}")

    model_class_path = os.path.join(base_dir, "xgb_flare_model.json")
    model_impact_path = os.path.join(base_dir, "xgb_impact_model.json")
    
    model_class.save_model(model_class_path)
    model_impact.save_model(model_impact_path)
    
    print(f"\nModels saved to {model_class_path} and {model_impact_path}")
    print("\n[OK] Complete pipeline executed successfully!")

if __name__ == "__main__":
    main()
