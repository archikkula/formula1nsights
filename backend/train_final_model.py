import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import json
import os

def calculate_sample_weights(df, current_season):
    """Calculate sample weights based on temporal relevance for fastest lap predictions"""
    print("Calculating temporal sample weights for fastest lap model...")
    
    weights = np.ones(len(df))
    
    # Get unique seasons and calculate weights
    seasons = df['season'].values
    unique_seasons = sorted(df['season'].unique())
    
    if len(unique_seasons) <= 1:
        print("  Only one season found, using uniform weights")
        return weights
    
    # Base weight calculation - recent seasons get higher weights
    season_weights = {}
    for i, season in enumerate(unique_seasons):
        if season == current_season:
            # Current season gets highest weight for future predictions
            season_weights[season] = 2.5
        else:
            # Exponential decay for older seasons
            years_back = current_season - season
            if years_back <= 1:
                season_weights[season] = 2.0
            elif years_back <= 2:
                season_weights[season] = 1.5
            elif years_back <= 3:
                season_weights[season] = 1.0
            else:
                season_weights[season] = 0.6  # Lower weight for very old data
    
    # Apply weights
    for season, weight in season_weights.items():
        season_mask = seasons == season
        weights[season_mask] = weight
    
    # Normalize weights so they sum to the original sample count
    weights = weights * len(weights) / weights.sum()
    
    print(f"  Season weights: {season_weights}")
    print(f"  Weight distribution: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
    
    return weights

def train_fastest_lap_model():
    """Train fastest lap model with categorical feature support and temporal weighting"""
    
    print("=== TRAINING FASTEST LAP MODEL WITH TEMPORAL WEIGHTING ===")
    
    # Load data
    df = pd.read_csv("data/predictor_features.csv")
    print(f"Loaded {len(df)} records")
    
    # Load feature info to get current season
    try:
        with open('data/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        current_season = feature_info.get('current_season', 2025)
        temporal_weighting = feature_info.get('temporal_weighting', {})
    except:
        current_season = 2025
        temporal_weighting = {}
    
    print(f"Current season: {current_season}")
    print(f"Temporal weighting enabled: {temporal_weighting.get('enabled', False)}")
    
    # Check if we have categorical features
    has_categorical = 'driver' in df.columns and 'circuitId' in df.columns
    print(f"Using categorical features: {has_categorical}")
    
    # Filter out rows with missing fastest_lap
    df_clean = df.dropna(subset=['fastest_lap']).copy()
    print(f"Records with fastest lap data: {len(df_clean)}")
    
    # Calculate temporal weights
    sample_weights = calculate_sample_weights(df_clean, current_season)
    
    # Exclude target and metadata columns
    exclude_cols = ["fastest_lap", "finish_pos", "season", "round"]
    
    # Handle driver column
    if 'driver' in df_clean.columns:
        exclude_cols.append('driver')  # We'll encode it separately
    
    # Handle circuitId column  
    if 'circuitId' in df_clean.columns:
        exclude_cols.append('circuitId')  # We'll encode it separately
        
    # Also exclude any 'circuit' column if it exists
    if 'circuit' in df_clean.columns:
        exclude_cols.append('circuit')
    
    # Remove excluded columns
    X = df_clean.drop(columns=[col for col in exclude_cols if col in df_clean.columns])
    
    # Handle categorical features if they exist
    encoders = {}
    categorical_features = []
    
    if has_categorical:
        print("Encoding categorical features...")
        
        # Encode driver
        if 'driver' in df_clean.columns:
            le_driver = LabelEncoder()
            X['driver'] = le_driver.fit_transform(df_clean['driver'].fillna('unknown').astype(str))
            encoders['driver'] = le_driver
            categorical_features.append('driver')
            print(f"  Encoded driver: {len(le_driver.classes_)} unique drivers")
        
        # Encode circuitId
        if 'circuitId' in df_clean.columns:
            le_circuit = LabelEncoder()
            X['circuitId'] = le_circuit.fit_transform(df_clean['circuitId'].fillna('unknown').astype(str))
            encoders['circuitId'] = le_circuit
            categorical_features.append('circuitId')
            print(f"  Encoded circuitId: {len(le_circuit.classes_)} unique circuits")
    
    # Get target variable
    y = df_clean["fastest_lap"].values
    
    print(f"✅ Training with {X.shape[1]} features (excluding: {exclude_cols})")
    print(f"✅ Training samples: {X.shape[0]}")
    print(f"✅ Features: {list(X.columns)}")
    
    # Handle any remaining non-numeric columns
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"Warning: Converting object column {col} to numeric")
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Fill any NaN values
    X = X.fillna(0)
    
    # Use temporal weighting - split by time for validation
    df_clean = df_clean.sort_values(['season', 'round']).reset_index(drop=True)
    
    # Temporal split for validation
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    weights_train = sample_weights[:split_idx]
    
    train_seasons = df_clean.iloc[:split_idx]['season'].unique()
    test_seasons = df_clean.iloc[split_idx:]['season'].unique()
    
    print(f"Training seasons: {sorted(train_seasons)}")
    print(f"Testing seasons: {sorted(test_seasons)}")
    print(f"Training with temporal weights: min={weights_train.min():.2f}, max={weights_train.max():.2f}")
    
    # Train model with sample weights
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model with temporal weighting...")
    model.fit(X_train, y_train, sample_weight=weights_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_spearman, _ = spearmanr(y_test, test_pred)
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test Spearman: {test_spearman:.3f}")
    
    # Retrain on full dataset for final model
    print("\nRetraining on full dataset...")
    final_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    final_model.fit(X, y, sample_weight=sample_weights)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    joblib.dump(final_model, "models/xgb_fastest_lap.pkl")
    print("✅ Saved model to models/xgb_fastest_lap.pkl")
    
    # Save encoders if we have categorical features
    if encoders:
        joblib.dump(encoders, "models/fastest_lap_encoders.pkl")
        print("✅ Saved encoders to models/fastest_lap_encoders.pkl")
    
    # Feature importance
    importance = final_model.feature_importances_
    feature_importance = list(zip(X.columns, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Feature Importance:")
    for feat, imp in feature_importance[:10]:
        print(f"  {feat}: {imp:.3f}")
    
    # Save model info
    model_info = {
        'model_type': 'fastest_lap',
        'feature_columns': list(X.columns),
        'n_features': len(X.columns),
        'categorical_features': categorical_features,
        'uses_categorical_encoding': has_categorical,
        'uses_temporal_weighting': True,
        'current_season': current_season,
        'training_samples': len(X),
        'excluded_columns': exclude_cols,
        'performance': {
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_spearman': float(test_spearman),
            'train_r2': float(train_r2)
        },
        'feature_importance': [(f, float(i)) for f, i in feature_importance[:20]],
        'temporal_weighting': {
            'enabled': True,
            'current_season_weight': 2.5,
            'description': 'Current season data weighted more heavily for fastest lap predictions'
        }
    }
    
    # Update or create feature info
    try:
        with open('data/feature_info.json', 'r') as f:
            existing_info = json.load(f)
    except FileNotFoundError:
        existing_info = {}
    
    existing_info['fastest_lap_model'] = model_info
    
    with open('data/feature_info.json', 'w') as f:
        json.dump(existing_info, f, indent=2)
    
    print("✅ Updated feature_info.json")
    
    # Verify model features
    print(f"✅ Model trained on features: {list(final_model.feature_names_in_)}")
    print(f"✅ Feature count: {len(final_model.feature_names_in_)}")
    
    return final_model, encoders

def train_enhanced_fastest_lap_with_current_season_features():
    """Enhanced fastest lap training that uses current season weighted features"""
    
    print("=== ENHANCED FASTEST LAP MODEL WITH CURRENT SEASON EMPHASIS ===")
    
    # Load data
    df = pd.read_csv("data/predictor_features.csv")
    print(f"Loaded {len(df)} records")
    
    # Load feature info
    try:
        with open('data/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        current_season = feature_info.get('current_season', 2025)
    except:
        current_season = 2025
    
    # Filter out rows with missing fastest_lap
    df_clean = df.dropna(subset=['fastest_lap']).copy()
    print(f"Records with fastest lap data: {len(df_clean)}")
    
    # Prioritize current season weighted features if available
    priority_features = [
        'weighted_career_mean', 'weighted_podium_rate',
        'current_season_mean', 'current_season_podium_rate'
    ]
    
    available_priority_features = [f for f in priority_features if f in df_clean.columns]
    print(f"Available current season weighted features: {available_priority_features}")
    
    # Calculate temporal weights with higher emphasis on current season for fastest lap
    sample_weights = calculate_sample_weights(df_clean, current_season)
    
    # Exclude target and metadata columns
    exclude_cols = ["fastest_lap", "finish_pos", "season", "round"]
    
    # Handle categorical features
    has_categorical = 'driver' in df_clean.columns and 'circuitId' in df_clean.columns
    encoders = {}
    categorical_features = []
    
    if has_categorical:
        print("Encoding categorical features...")
        
        # Encode driver
        if 'driver' in df_clean.columns:
            le_driver = LabelEncoder()
            df_clean['driver_encoded'] = le_driver.fit_transform(df_clean['driver'].fillna('unknown').astype(str))
            encoders['driver'] = le_driver
            categorical_features.append('driver')
            exclude_cols.append('driver')  # Remove original, keep encoded
            print(f"  Encoded driver: {len(le_driver.classes_)} unique drivers")
        
        # Encode circuitId
        if 'circuitId' in df_clean.columns:
            le_circuit = LabelEncoder()
            df_clean['circuitId_encoded'] = le_circuit.fit_transform(df_clean['circuitId'].fillna('unknown').astype(str))
            encoders['circuitId'] = le_circuit
            categorical_features.append('circuitId')
            exclude_cols.append('circuitId')  # Remove original, keep encoded
            print(f"  Encoded circuitId: {len(le_circuit.classes_)} unique circuits")
    
    # Remove excluded columns and handle any 'circuit' column
    if 'circuit' in df_clean.columns:
        exclude_cols.append('circuit')
    
    X = df_clean.drop(columns=[col for col in exclude_cols if col in df_clean.columns])
    y = df_clean["fastest_lap"].values
    
    # Handle any remaining non-numeric columns
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"Warning: Converting object column {col} to numeric")
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Fill any NaN values
    X = X.fillna(0)
    
    print(f"✅ Enhanced training with {X.shape[1]} features")
    print(f"✅ Priority features included: {[f for f in available_priority_features if f in X.columns]}")
    
    # Temporal validation split
    df_clean = df_clean.sort_values(['season', 'round']).reset_index(drop=True)
    split_idx = int(0.8 * len(X))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    weights_train = sample_weights[:split_idx]
    
    # Enhanced model with temporal weighting
    model = xgb.XGBRegressor(
        n_estimators=200,  # More trees for better current season learning
        max_depth=6,       # Slightly deeper for complex interactions
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training enhanced model with current season emphasis...")
    model.fit(X_train, y_train, sample_weight=weights_train)
    
    # Evaluate
    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_spearman, _ = spearmanr(y_test, test_pred)
    
    print(f"\nENHANCED MODEL PERFORMANCE:")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test Spearman: {test_spearman:.3f}")
    
    # Final model on all data
    final_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    final_model.fit(X, y, sample_weight=sample_weights)
    
    return final_model, encoders, X.columns.tolist(), test_r2, test_rmse, test_spearman

if __name__ == "__main__":
    print("Choose training approach:")
    print("1. Standard fastest lap model with temporal weighting")
    print("2. Enhanced model with current season emphasis")
    
    try:
        choice = input("Enter choice (1 or 2, default 2): ").strip()
        if choice == "1":
            model, encoders = train_fastest_lap_model()
        else:
            model, encoders, features, test_r2, test_rmse, test_spearman = train_enhanced_fastest_lap_with_current_season_features()
            
            # Save enhanced model
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/xgb_fastest_lap.pkl")
            
            if encoders:
                joblib.dump(encoders, "models/fastest_lap_encoders.pkl")
            
            # Update feature info with enhanced model info
            try:
                with open('data/feature_info.json', 'r') as f:
                    existing_info = json.load(f)
            except FileNotFoundError:
                existing_info = {}
            
            existing_info['fastest_lap_model'] = {
                'model_type': 'fastest_lap_enhanced',
                'feature_columns': features,
                'n_features': len(features),
                'categorical_features': list(encoders.keys()) if encoders else [],
                'uses_categorical_encoding': bool(encoders),
                'uses_temporal_weighting': True,
                'uses_current_season_emphasis': True,
                'performance': {
                    'test_r2': float(test_r2),
                    'test_rmse': float(test_rmse),
                    'test_spearman': float(test_spearman)
                },
                'enhanced_features': True
            }
            
            with open('data/feature_info.json', 'w') as f:
                json.dump(existing_info, f, indent=2)
            
            print("✅ Saved enhanced model to models/xgb_fastest_lap.pkl")
            
    except (KeyboardInterrupt, EOFError):
        # Default to enhanced approach
        model, encoders, features, test_r2, test_rmse, test_spearman = train_enhanced_fastest_lap_with_current_season_features()
        
        # Save models
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/xgb_fastest_lap.pkl")
        
        if encoders:
            joblib.dump(encoders, "models/fastest_lap_encoders.pkl")
        
        print("✅ Enhanced fastest lap model training complete!")
    
    print("\n" + "="*60)
    print("FASTEST LAP MODEL TRAINING COMPLETE WITH TEMPORAL WEIGHTING")
    print("="*60)
    print("✅ Model emphasizes current season data for better predictions")
    print("✅ Temporal weighting applied to training samples")
    print("✅ Enhanced for current season and future race predictions")
    print("✅ Compatible with enhanced prediction API")