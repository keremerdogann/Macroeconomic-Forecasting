import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import (load_and_clean_data, add_global_features, 
                               handle_missing_values, detect_and_cap_outliers, 
                               feature_engineering, scale_features)
from src.modeling import get_models, evaluate_models, plot_feature_importance

# Data path (updated to point to the Excel file)
DATA_PATH = os.path.join("data", "raw", "main_data.xlsx")

def main():
    print("Starting Macroeconomic Forecasting Pipeline...\n")
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found. Please ensure your file is located at: {DATA_PATH}")
        return

    # 1. Advanced Preprocessing
    df = load_and_clean_data(DATA_PATH)
    df = add_global_features(df)
    df, numeric_cols = handle_missing_values(df)
    df = detect_and_cap_outliers(df, numeric_cols)
    df = feature_engineering(df)
    
    # 2. X and y Separation (Target: GDP_Growth)
    y = df['GDP_Growth']
    
    # Feature Elimination (FDI was proven to be noisy in research, removing it)
    cols_to_drop = ['KEY', 'KOD', 'Ülke', 'Yıl', 'GDP_Growth', 'FDI']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 5. Modeling
    models = get_models()
    results = evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 6. Dynamically select the best model based on highest R2 score
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_r2 = results[best_model_name]['R2']
    
    print(f"\n🏆 Best Model Automatically Selected: {best_model_name} (R2: {best_r2:.4f})")
    
    best_model = models[best_model_name]
    plot_feature_importance(best_model, X.columns)
    
    # Save the best model
    model_dir = os.path.join("models", "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{best_model_name.replace(' ', '_').lower()}_best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest model successfully saved to: {model_path}")

if __name__ == "__main__":
    main()