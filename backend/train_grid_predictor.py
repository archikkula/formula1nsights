import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for grid position prediction"""
    print("Loading data for grid position prediction...")
    
    # Load the features
    df = pd.read_csv("data/predictor_features.csv")
    print(f"Loaded {len(df)} records")
    
    # Filter out records without grid positions (target variable)
    df = df.dropna(subset=['grid_position'])
    print(f"After filtering: {len(df)} records with grid positions")
    
    return df

def prepare_pre_qualifying_features(df):
    """Prepare features that are available BEFORE qualifying"""
    print("Preparing pre-qualifying features...")
    
    # Features available before qualifying (no Q1, Q2, Q3, grid_position)
    pre_qual_features = [
        'leak_proof_weighted_mean', 'leak_proof_weighted_podium_rate',
        'leak_proof_current_mean', 'leak_proof_current_podium_rate', 
        'leak_proof_current_wins_rate', 'leak_proof_recent_form', 
        'leak_proof_consistency', 'leak_proof_peak_performance',
        'leak_proof_career_mean', 'leak_proof_career_podium', 
        'leak_proof_last3_fin', 'leak_proof_points_rate', 
        'leak_proof_team_mean', 'leak_proof_team_podium_rate',
        'leak_proof_circuit_mean_norm', 'leak_proof_circuit_podium_norm',
        'length_km', 'corners', 'laps', 'avg_temp', 'precip_mm',
        'driver', 'circuitId', 'team'
    ]
    
    # Keep only available features
    available_features = [f for f in pre_qual_features if f in df.columns]
    print(f"Using {len(available_features)} pre-qualifying features")
    
    return df[available_features + ['grid_position']], available_features

def encode_categorical_features(df, categorical_features):
    """Encode categorical features"""
    print("Encoding categorical features...")
    
    encoders = {}
    df_encoded = df.copy()
    
    for feature in categorical_features:
        if feature in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
            encoders[feature] = le
            print(f"  Encoded {feature}: {len(le.classes_)} classes")
    
    return df_encoded, encoders

def train_grid_position_model(X, y, feature_names):
    """Train XGBoost model to predict grid positions"""
    print("Training grid position prediction model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"CV R² (mean): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    feature_importance = list(zip(feature_names, [float(imp) for imp in model.feature_importances_]))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Feature Importance:")
    for feature, importance in feature_importance[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    return model, {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'feature_importance': feature_importance
    }

def save_grid_model_and_info(model, encoders, feature_names, metrics, categorical_features):
    """Save the grid position model and metadata"""
    print("Saving grid position model...")
    
    # Save model
    joblib.dump(model, "models/xgb_grid_predictor.pkl")
    print("✅ Saved models/xgb_grid_predictor.pkl")
    
    # Save encoders
    joblib.dump(encoders, "models/label_encoders_grid.pkl")
    print("✅ Saved models/label_encoders_grid.pkl")
    
    # Create grid model info
    grid_model_info = {
        'model_type': 'xgboost_regressor',
        'target': 'grid_position',
        'feature_columns': feature_names,
        'categorical_features': categorical_features,
        'n_features': len(feature_names),
        'train_r2': metrics['train_r2'],
        'test_r2': metrics['test_r2'],
        'train_rmse': metrics['train_rmse'],
        'test_rmse': metrics['test_rmse'],
        'cv_r2_mean': metrics['cv_r2_mean'],
        'cv_r2_std': metrics['cv_r2_std'],
        'feature_importance': metrics['feature_importance'][:20]  # Top 20
    }
    
    # Load existing model_info and ADD grid model info (don't replace)
    model_info_path = "data/model_info.json"
    try:
        with open(model_info_path, 'r') as f:
            existing_info = json.load(f)
        print("✅ Loaded existing model_info.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load existing model_info.json: {e}")
        print("Creating new model_info.json")
        existing_info = {}
    
    # ADD grid model info to existing info (preserve everything else)
    existing_info['grid_position_model'] = grid_model_info
    
    # Save updated model info
    with open(model_info_path, 'w') as f:
        json.dump(existing_info, f, indent=2)
    
    print("✅ Added grid model info to data/model_info.json (preserved existing data)")
    print(f"   Grid model R²: {metrics['test_r2']:.4f}")
    print(f"   Grid model RMSE: {metrics['test_rmse']:.4f}")


def main():
    print("=" * 60)
    print("TRAINING GRID POSITION PREDICTOR")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    df_features, feature_names = prepare_pre_qualifying_features(df)
    
    # Define categorical features
    categorical_features = ['driver', 'circuitId', 'team']
    
    # Encode categorical features
    df_encoded, encoders = encode_categorical_features(df_features, categorical_features)
    
    # Prepare X and y
    X = df_encoded.drop('grid_position', axis=1)
    y = df_encoded['grid_position']
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f}")
    
    # Train model
    model, metrics = train_grid_position_model(X, y, feature_names)
    
    # Save everything
    save_grid_model_and_info(model, encoders, feature_names, metrics, categorical_features)
    
    print("\n" + "=" * 60)
    print("GRID POSITION PREDICTOR TRAINING COMPLETE!")
    print("=" * 60)
    print("✅ Model saved: models/xgb_grid_predictor.pkl")
    print("✅ Encoders saved: models/label_encoders_grid.pkl")
    print("✅ Info updated: data/model_info.json")
    print(f"\n📊 Performance Summary:")
    print(f"   Test R²: {metrics['test_r2']:.4f}")
    print(f"   Test RMSE: {metrics['test_rmse']:.4f} positions")
    print(f"   CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print("\n🎯 Ready for two-stage prediction pipeline!")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    main()