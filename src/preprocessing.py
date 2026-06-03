import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest

def load_and_clean_data(filepath):
    print(f"Loading data: {filepath}")
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # Clean string missing values
    df.replace(['EKSİK VERİ', ' '], np.nan, inplace=True)
    
    # Rename columns to English standardized names if they are in Turkish
    rename_map = {
        'GFCF (Yatırım)': 'GFCF', 'FDI (Yabancı Yatırım)': 'FDI',
        'CPI (Enflasyon)': 'CPI', 'Unemployment (İşsizlik)': 'Unemployment',
        'Current_Account (Cari Denge)': 'Current_Account', 'Exchange_Rate (Kur)': 'Exchange_Rate',
        'Govt_Exp (Devlet Harcaması)': 'Govt_Exp', 'Pop_Growth (Nüfus Artışı)': 'Pop_Growth',
        'Export_GDP (İhracat)': 'Export_GDP', 'GDP_Growth (TARGET - Tahmin edilecek sütun)': 'GDP_Growth'
    }
    df = df.rename(columns=rename_map)
    
    # 1. Time Filter (1990 onwards)
    if 'Yıl' in df.columns:
        df = df[df['Yıl'] >= 1990]
        
    # 2. Target Clean
    df = df.dropna(subset=['GDP_Growth'])
    
    # Ensure numeric columns
    numeric_cols = ['GFCF', 'FDI', 'CPI', 'Unemployment', 'Current_Account',
                    'Exchange_Rate', 'Govt_Exp', 'Pop_Growth', 'Export_GDP', 'GDP_Growth']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 3. Drop countries with >50% missing data
    if 'Ülke' in df.columns:
        countries_to_drop = []
        for country in df['Ülke'].unique():
            country_data = df[df['Ülke'] == country]
            missing_ratio = country_data[numeric_cols].isnull().sum().sum() / country_data[numeric_cols].size
            if missing_ratio > 0.5:
                countries_to_drop.append(country)
        df = df[~df['Ülke'].isin(countries_to_drop)]
        
    return df

def add_global_features(df):
    print("Integrating global macroeconomic features...")
    global_data = {
        'Yıl': list(range(1990, 2025)),
        'Brent_Oil': [23.7, 20.0, 19.3, 17.0, 15.8, 17.0, 20.7, 19.1, 12.7, 17.9,
                      28.5, 24.4, 25.0, 28.8, 38.3, 54.5, 65.1, 72.3, 97.2, 61.6,
                      79.5, 111.2, 111.6, 108.6, 98.9, 52.3, 43.7, 54.1, 71.3, 64.3,
                      41.8, 70.9, 100.9, 82.5, 80.0],
        'FED_Rate': [8.1, 5.7, 3.5, 3.0, 4.2, 5.8, 5.3, 5.5, 5.3, 4.9,
                     6.2, 3.8, 1.6, 1.1, 1.3, 3.2, 4.9, 5.0, 1.9, 0.1,
                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 1.0, 1.8, 2.1,
                     0.3, 0.1, 1.6, 5.0, 5.3],
        'VIX_Index': [23, 18, 15, 12, 13, 12, 16, 25, 25, 24,
                      23, 25, 27, 21, 15, 12, 12, 17, 32, 31,
                      20, 24, 17, 14, 14, 16, 13, 11, 16, 15,
                      29, 19, 25, 16, 14],
        'Global_GDP_Growth': [2.8, 1.5, 2.1, 1.8, 2.8, 3.1, 3.3, 3.6, 2.6, 3.1,
                              4.4, 1.9, 2.2, 2.9, 4.4, 3.9, 4.3, 4.3, 2.1, -1.3,
                              4.5, 3.3, 2.7, 2.8, 3.0, 3.0, 2.6, 3.3, 3.2, 2.5,
                              -3.0, 6.2, 3.0, 2.9, 2.7]
    }
    df_global = pd.DataFrame(global_data)
    if 'Yıl' in df.columns:
        df = pd.merge(df, df_global, on='Yıl', how='left')
    return df

def handle_missing_values(df):
    print("Applying Hybrid Imputation (Linear Interpolation + MICE)...")
    if 'Ülke' in df.columns and 'Yıl' in df.columns:
        df = df.sort_values(by=['Ülke', 'Yıl'])
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Yıl', 'KEY', 'KOD'], errors='ignore').tolist()
    
    # 1. Linear Interpolation (Time-Series integrity)
    if 'Ülke' in df.columns:
        df[numeric_cols] = df.groupby('Ülke')[numeric_cols].transform(
            lambda group: group.interpolate(method='linear', limit_direction='both')
        )
        
    # 2. Iterative Imputer (MICE)
    iter_imputer = IterativeImputer(max_iter=10, random_state=42)
    df[numeric_cols] = iter_imputer.fit_transform(df[numeric_cols])
    
    return df, numeric_cols

def detect_and_cap_outliers(df, numeric_cols):
    print("Detecting and capping outliers using Isolation Forest...")
    cols_for_outlier = [c for c in numeric_cols if c != 'GDP_Growth']
    
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['Outlier_ISO'] = iso_forest.fit_predict(df[cols_for_outlier])
    
    # Winsorization (Baskılama)
    for col in cols_for_outlier:
        lower = df.loc[df['Outlier_ISO'] == 1, col].min()
        upper = df.loc[df['Outlier_ISO'] == 1, col].max()
        
        df.loc[(df['Outlier_ISO'] == -1) & (df[col] < lower), col] = lower
        df.loc[(df['Outlier_ISO'] == -1) & (df[col] > upper), col] = upper
        
    df = df.drop(columns=['Outlier_ISO'])
    return df

def feature_engineering(df):
    print("Performing Feature Engineering (Lag features, Log transform)...")
    # Log Transform for Absolute values
    if 'GFCF' in df.columns:
        df['Log_GFCF'] = np.log1p(df['GFCF'])
        df = df.drop(columns=['GFCF']) 
        
    # Time-Series Lag Features (t-1)
    if 'Ülke' in df.columns and 'Yıl' in df.columns:
        df = df.sort_values(by=['Ülke', 'Yıl'])
        df['GDP_Growth_Lag1'] = df.groupby('Ülke')['GDP_Growth'].shift(1)
        df['CPI_Lag1'] = df.groupby('Ülke')['CPI'].shift(1)
        df['Unemployment_Lag1'] = df.groupby('Ülke')['Unemployment'].shift(1)
        
        # Drop rows with NA caused by the lag shift
        df = df.dropna(subset=['GDP_Growth_Lag1'])
        
    return df

def scale_features(X_train, X_test):
    print("Scaling data (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler