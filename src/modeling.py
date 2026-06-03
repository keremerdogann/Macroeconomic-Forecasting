from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_models():
    """Returns the models to be used with optimized hyperparameters."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            n_jobs=-1, 
            random_state=42
        )
    }

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Trains and evaluates all models on the test set."""
    results = {}
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"[{name}] R2: {r2:.4f} | RMSE: {rmse:.4f}")
        
    return results

def plot_feature_importance(model, feature_names, top_n=10):
    """Plots feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        print("This model does not support feature_importance.")
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title('Top Feature Importances (Macroeconomic Drivers)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'.")