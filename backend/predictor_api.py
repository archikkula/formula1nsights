from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import numpy as np
import json
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

bp = Blueprint("predictor_api", __name__)

# Global model variables
fastest_model = None
finish_model = None
finish_model_pre_qual = None
FEATURE_COLUMNS = None
PRE_QUAL_FEATURES = None
CATEGORICAL_FEATURES = None
ENCODERS_PRE = None
ENCODERS_POST = None
MODEL_INFO = None

def load_models():
    """Load models and encoders with data/ folder structure support"""
    global fastest_model, finish_model, finish_model_pre_qual
    global FEATURE_COLUMNS, PRE_QUAL_FEATURES, CATEGORICAL_FEATURES
    global ENCODERS_PRE, ENCODERS_POST, MODEL_INFO
    
    try:
        # Load existing models
        fastest_model = joblib.load("models/xgb_fastest_lap.pkl")
        finish_model = joblib.load("models/xgb_finish_ranker.pkl")
        
        # Try to load the new pre-qualifying model
        try:
            finish_model_pre_qual = joblib.load("models/xgb_finish_ranker_pre_qual.pkl")
            print("✅ Loaded pre-qualifying model for future predictions")
        except FileNotFoundError:
            print("⚠️  Pre-qualifying model not found - using post-qualifying model for all predictions")
            finish_model_pre_qual = finish_model
        
        # Load categorical encoders if available
        try:
            ENCODERS_PRE = joblib.load("models/label_encoders_pre_qual.pkl")
            ENCODERS_POST = joblib.load("models/label_encoders_post_qual.pkl")
            print("✅ Loaded categorical encoders")
        except FileNotFoundError:
            print("⚠️  No categorical encoders found - using legacy mode")
            ENCODERS_PRE = {}
            ENCODERS_POST = {}
        
        # Load enhanced model info from data/ folder structure
        model_info_paths = [
            'data/model_info.json',
            'data/feature_info.json',
            'feature_info.json'
        ]
        
        MODEL_INFO = None
        for path in model_info_paths:
            try:
                with open(path, 'r') as f:
                    MODEL_INFO = json.load(f)
                print(f"✅ Loaded model info from {path}")
                break
            except FileNotFoundError:
                continue
        
        if MODEL_INFO is None:
            print("⚠️  No model info found - using defaults")
            MODEL_INFO = {}
        
        # Handle both old and new model info formats
        if 'pre_qualifying_model' in MODEL_INFO:
            # New enhanced format with categorical support
            FEATURE_COLUMNS = MODEL_INFO.get('feature_columns', [])
            PRE_QUAL_FEATURES = MODEL_INFO.get('pre_qual_features', FEATURE_COLUMNS)
            CATEGORICAL_FEATURES = MODEL_INFO.get('categorical_features', [])
        else:
            # Old format - backward compatibility
            FEATURE_COLUMNS = MODEL_INFO.get('feature_columns', [])
            PRE_QUAL_FEATURES = FEATURE_COLUMNS
            CATEGORICAL_FEATURES = MODEL_INFO.get('categorical_features', [])
        
        print(f"✅ Loaded models with {len(FEATURE_COLUMNS)} post-qual features, {len(PRE_QUAL_FEATURES)} pre-qual features")
        print(f"✅ Categorical features: {CATEGORICAL_FEATURES}")
        print(f"✅ Data structure: {'organized' if MODEL_INFO.get('data_structure', {}).get('base_path') else 'legacy'}")
        
        # Debug: Print expected feature order for pre-qualifying model
        if 'pre_qualifying_model' in MODEL_INFO and 'feature_columns' in MODEL_INFO['pre_qualifying_model']:
            expected_features = MODEL_INFO['pre_qualifying_model']['feature_columns']
            print(f"✅ Pre-qual model expects {len(expected_features)} features in specific order")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

def prepare_features_enhanced(df, feature_columns, encoders, categorical_features, model_type="pre_qualifying"):
    """Enhanced feature preparation with proper categorical encoding and feature ordering"""
    print(f"Preparing features for {model_type} model...")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Expected features: {len(feature_columns)}")
    print(f"Available columns: {list(df.columns)}")
    
    # Create a copy to avoid modifying original
    features_df = df.copy()
    
    # First, encode categorical features
    for cat_feature in categorical_features:
        if cat_feature in features_df.columns and cat_feature in encoders:
            le = encoders[cat_feature]
            print(f"Encoding {cat_feature} with {len(le.classes_)} classes")
            
            def safe_encode(x):
                try:
                    return le.transform([str(x)])[0]
                except ValueError:
                    # Unknown category - use first class or 0
                    if len(le.classes_) > 0:
                        return le.transform([le.classes_[0]])[0]
                    else:
                        return 0
            
            features_df[cat_feature] = features_df[cat_feature].apply(safe_encode)
        elif cat_feature in features_df.columns:
            print(f"Warning: No encoder found for {cat_feature}, using label encoding")
            # Fallback: simple label encoding
            unique_vals = features_df[cat_feature].unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            features_df[cat_feature] = features_df[cat_feature].map(mapping).fillna(0)
    
    # Create feature matrix with correct column order
    final_features_df = pd.DataFrame()
    
    for col in feature_columns:
        if col in features_df.columns:
            final_features_df[col] = features_df[col]
        else:
            # Handle missing features with smart defaults
            print(f"Missing feature {col}, using default value")
            if col in categorical_features:
                final_features_df[col] = 0  # Encoded categorical default
            elif any(x in col.lower() for x in ["mean", "rate", "podium"]):
                final_features_df[col] = 0.5
            elif col in ["Q1", "Q2", "Q3"]:
                final_features_df[col] = 90.0
            elif "grid" in col or "position" in col:
                final_features_df[col] = 10.0
            elif "temp" in col:
                final_features_df[col] = 22.0
            elif "precip" in col:
                final_features_df[col] = 0.0
            elif col in ["made_Q2", "made_Q3"]:
                final_features_df[col] = 0
            elif any(x in col.lower() for x in ["length", "corners", "laps"]):
                if "length" in col:
                    final_features_df[col] = 5.0
                elif "corners" in col:
                    final_features_df[col] = 15
                elif "laps" in col:
                    final_features_df[col] = 60
            elif "ratio" in col or "norm" in col:
                final_features_df[col] = 0.0  # Interaction features default
            else:
                final_features_df[col] = 0.0
    
    # Fill any remaining NaN values
    final_features_df = final_features_df.fillna(0.0)
    
    print(f"Final feature matrix shape: {final_features_df.shape}")
    print(f"Feature order matches model: {list(final_features_df.columns) == feature_columns}")
    
    return final_features_df

def get_prediction_confidence(model, X, n_samples=20):
    """Generate prediction confidence using bootstrap sampling"""
    if len(X) == 0:
        return [], []
    
    predictions = []
    for _ in range(n_samples):
        # Add small random noise to simulate uncertainty
        X_noise = X.copy()
        for col in X_noise.columns:
            if X_noise[col].dtype in ['float64', 'int64']:
                noise = np.random.normal(0, 0.02 * X_noise[col].std(), len(X_noise))
                X_noise[col] += noise
        
        pred = model.predict(X_noise)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    pred_std = np.std(predictions, axis=0)
    pred_mean = np.mean(predictions, axis=0)
    
    return pred_mean, pred_std

@bp.route("/predict_finish_pos_round")
def predict_finish_pos_round():
    """Predict finish positions for a completed race (with qualifying data)"""
    try:
        season = request.args.get("season", type=int)
        rnd = request.args.get("round", type=int)
        include_confidence = request.args.get("confidence", "false").lower() == "true"
        
        # Try to load data from organized data/ folder structure
        data_paths = [
            "data/predictor_features.csv",
            "predictor_features.csv"
        ]
        
        feats = None
        for path in data_paths:
            try:
                feats = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if feats is None:
            return jsonify({"error": "predictor_features.csv not found - run feature engineering first"}), 404
        
        sub = feats.loc[(feats["season"] == season) & (feats["round"] == rnd)]
        
        if sub.empty:
            return jsonify({"error": "No features for that season/round"}), 404
        
        drivers = sub["driver"].tolist() if "driver" in sub.columns else ["unknown"] * len(sub)
        
        # Use enhanced feature preparation
        X = prepare_features_enhanced(sub, FEATURE_COLUMNS, ENCODERS_POST, CATEGORICAL_FEATURES, "post_qualifying")
        
        if include_confidence:
            scores, confidence = get_prediction_confidence(finish_model, X)
        else:
            scores = finish_model.predict(X)
            confidence = None
        
        # Convert scores to positions
        n_drivers = len(scores)
        order = np.argsort(scores)
        predicted_positions = np.empty_like(order, dtype=float)
        predicted_positions[order] = np.arange(1, n_drivers + 1)
        
        actuals = sub["finish_pos"].tolist() if "finish_pos" in sub.columns else [None] * n_drivers
        
        results = []
        for i, (d, p, a) in enumerate(zip(drivers, predicted_positions, actuals)):
            entry = {
                "driver": d, 
                "predicted": float(p),
                "model_score": float(scores[i])
            }
            if a is not None:
                entry["actual"] = float(a)
            if confidence is not None:
                entry["confidence"] = float(confidence[i])
            results.append(entry)
        
        results.sort(key=lambda x: x["predicted"])
        
        return jsonify({
            "season": season, 
            "round": rnd, 
            "predictions": results, 
            "total_drivers": n_drivers,
            "model_type": "post_qualifying",
            "uses_categorical": bool(CATEGORICAL_FEATURES),
            "data_source": "organized" if MODEL_INFO.get('data_structure', {}).get('base_path') else "legacy"
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@bp.route("/predict_fastest_lap_round")
def predict_fastest_lap_round():
    """Predict fastest lap times for a completed race"""
    try:
        season = request.args.get("season", type=int)
        rnd = request.args.get("round", type=int)
        include_confidence = request.args.get("confidence", "false").lower() == "true"
        
        # Try to load data from data/ folder
        data_paths = [
            "data/predictor_features.csv",
            "predictor_features.csv"
        ]
        
        feats = None
        for path in data_paths:
            try:
                feats = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if feats is None:
            return jsonify({"error": "predictor_features.csv not found - run feature engineering first"}), 404
        
        sub = feats.loc[(feats["season"] == season) & (feats["round"] == rnd)]
        
        if sub.empty:
            return jsonify({"error": "No features for that season/round"}), 404
        
        drivers = sub["driver"].tolist() if "driver" in sub.columns else ["unknown"] * len(sub)
        
        # Use enhanced feature preparation
        X = prepare_features_enhanced(sub, FEATURE_COLUMNS, ENCODERS_POST, CATEGORICAL_FEATURES, "post_qualifying")
        
        if include_confidence:
            preds, confidence = get_prediction_confidence(fastest_model, X)
        else:
            preds = fastest_model.predict(X)
            confidence = None
        
        actuals = sub["fastest_lap"].tolist() if "fastest_lap" in sub.columns else [None] * len(sub)
        
        results = []
        for i, (d, p, a) in enumerate(zip(drivers, preds, actuals)):
            entry = {
                "driver": d, 
                "predicted": float(p)
            }
            if a is not None:
                entry["actual"] = float(a)
            if confidence is not None:
                entry["confidence"] = float(confidence[i])
            results.append(entry)
        
        return jsonify({
            "season": season, 
            "round": rnd, 
            "predictions": results,
            "model_type": "fastest_lap",
            "uses_categorical": bool(CATEGORICAL_FEATURES)
        })
        
    except Exception as e:
        return jsonify({"error": f"Fastest lap prediction failed: {str(e)}"}), 500

@bp.route("/predict_finish_pos_future")
def predict_finish_pos_future():
    try:
        season = request.args.get("season", type=int)
        round_num = request.args.get("round", type=int)
        if season is None:
            from datetime import datetime
            current_date = datetime.now()
            season = current_date.year if current_date.month > 2 else current_date.year - 1

        use_pre_qual = request.args.get("pre_qualifying", "true").lower() == "true"
        include_confidence = request.args.get("confidence", "false").lower() == "true"

        up_paths = [
            "data/upcoming_features.csv",
            "upcoming_features.csv"
        ]

        up = None
        for path in up_paths:
            try:
                up = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue

        if up is None:
            return jsonify({
                "error": f"upcoming_features.csv not found",
                "suggestion": "Run the automated feature engineering to generate future race features"
            }), 404

        sub = up.loc[up["season"] == season].copy() if "season" in up.columns else up.copy()

        # Filter by round if specified
        if round_num is not None and "round" in sub.columns:
            sub = sub[sub["round"] == round_num]

        if sub.empty:
            return jsonify({
                "error": f"No upcoming races for season {season}" + (f", round {round_num}" if round_num else ""),
                "suggestion": f"Try a different round or season."
            }), 404

        # Extract driver information
        if "driver" in sub.columns:
            drivers = sub["driver"].tolist()
        else:
            drivers = ["unknown"] * len(sub)
            sub['driver'] = drivers

        # Select model and features
        if use_pre_qual and finish_model_pre_qual is not None:
            model = finish_model_pre_qual
            features = PRE_QUAL_FEATURES
            encoders = ENCODERS_PRE
            model_type = "pre_qualifying"
        else:
            model = finish_model
            features = FEATURE_COLUMNS
            encoders = ENCODERS_POST
            model_type = "post_qualifying"

        print(f"Using {model_type} model with {len(features)} features")
        
        # Use enhanced feature preparation with proper ordering
        X = prepare_features_enhanced(sub, features, encoders, CATEGORICAL_FEATURES, model_type)

        if include_confidence:
            scores, confidence = get_prediction_confidence(model, X)
        else:
            scores = model.predict(X)
            confidence = None

        sub['pred_scores'] = scores

        results = []
        rounds = sub['round'].unique() if 'round' in sub.columns else [1]

        for round_id in sorted(rounds):
            if 'round' in sub.columns:
                round_data = sub[sub['round'] == round_id].copy()
            else:
                round_data = sub.copy()
                round_id = 1

            round_scores = round_data['pred_scores'].values
            round_drivers = round_data['driver'].tolist() if 'driver' in round_data.columns else drivers
            round_confidence = confidence[round_data.index] if confidence is not None and 'round' in sub.columns else confidence

            order = np.argsort(round_scores)
            predicted_positions = np.empty_like(order, dtype=float)
            predicted_positions[order] = np.arange(1, len(order) + 1)

            for i, driver in enumerate(round_drivers):
                entry = {
                    "round": int(round_id),
                    "driver": driver,
                    "predicted": float(predicted_positions[i]),
                    "model_score": float(round_scores[i])
                }
                if round_confidence is not None:
                    entry["confidence"] = float(round_confidence[i] if len(round_confidence) > i else 0.0)
                results.append(entry)

        return jsonify({
            "season": season,
            "predictions": results,
            "model_type": model_type,
            "total_predictions": len(results),
            "uses_categorical": bool(CATEGORICAL_FEATURES)
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Future prediction error: {error_details}")
        return jsonify({"error": f"Future finish pos prediction failed: {str(e)}", "details": error_details}), 500

@bp.route("/predict_fastest_lap_future")
def predict_fastest_lap_future():
    """Enhanced fastest lap predictions for future races"""
    try:
        season = request.args.get("season", type=int)
        include_confidence = request.args.get("confidence", "false").lower() == "true"
        
        # Try to load upcoming features from data/ folder
        up_paths = [
            "data/upcoming_features.csv",
            "upcoming_features.csv"
        ]
        
        df_up = None
        for path in up_paths:
            try:
                df_up = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if df_up is None:
            return jsonify({
                "error": "upcoming_features.csv not found",
                "suggestion": "Run feature engineering to generate future race features"
            }), 404
        
        mask = df_up["season"] == season if "season" in df_up.columns else pd.Series([True] * len(df_up))
        
        if not mask.any():
            return jsonify({"error": "No upcoming races for that season"}), 404
        
        subset = df_up.loc[mask].copy()
        
        # Extract driver information
        if "driver" in subset.columns:
            drivers = subset["driver"].tolist()
        else:
            drivers = ["unknown"] * len(subset)
        
        # Use enhanced feature preparation
        X = prepare_features_enhanced(subset, FEATURE_COLUMNS, ENCODERS_POST, CATEGORICAL_FEATURES, "post_qualifying")
        
        if include_confidence:
            preds, confidence = get_prediction_confidence(fastest_model, X)
        else:
            preds = fastest_model.predict(X)
            confidence = None
        
        rounds = subset["round"].tolist() if "round" in subset.columns else [1] * len(subset)
        
        predictions = []
        for i, (r, d, p) in enumerate(zip(rounds, drivers, preds)):
            entry = {
                "round": int(r), 
                "driver": d, 
                "predicted": float(p)
            }
            if confidence is not None:
                entry["confidence"] = float(confidence[i])
            predictions.append(entry)
        
        return jsonify({
            "season": season,
            "predictions": predictions,
            "model_type": "fastest_lap",
            "uses_categorical": bool(CATEGORICAL_FEATURES)
        })
        
    except Exception as e:
        return jsonify({"error": f"Future fastest lap prediction failed: {str(e)}"}), 500

@bp.route("/model_info")
def get_model_info():
    """Get information about loaded models and their performance"""
    try:
        info = {
            "models_loaded": {
                "fastest_lap": fastest_model is not None,
                "finish_position": finish_model is not None,
                "pre_qualifying": finish_model_pre_qual is not None
            },
            "feature_counts": {
                "post_qualifying": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0,
                "pre_qualifying": len(PRE_QUAL_FEATURES) if PRE_QUAL_FEATURES else 0
            },
            "uses_categorical_encoding": bool(CATEGORICAL_FEATURES),
            "categorical_features": CATEGORICAL_FEATURES if CATEGORICAL_FEATURES else [],
            "data_structure": MODEL_INFO.get('data_structure', {}) if MODEL_INFO else {},
            "current_season": MODEL_INFO.get('current_season') if MODEL_INFO else None,
            "season_complete": MODEL_INFO.get('season_complete') if MODEL_INFO else None
        }
        
        if MODEL_INFO:
            # Add performance metrics if available
            if 'pre_qualifying_model' in MODEL_INFO:
                info["performance"] = {
                    "pre_qualifying": {
                        "test_r2": MODEL_INFO['pre_qualifying_model'].get('test_r2'),
                        "test_rmse": MODEL_INFO['pre_qualifying_model'].get('test_rmse'),
                        "avg_cv_r2": MODEL_INFO['pre_qualifying_model'].get('avg_cv_r2')
                    },
                    "post_qualifying": {
                        "test_r2": MODEL_INFO['post_qualifying_model'].get('test_r2'),
                        "test_rmse": MODEL_INFO['post_qualifying_model'].get('test_rmse'),
                        "avg_cv_r2": MODEL_INFO['post_qualifying_model'].get('avg_cv_r2')
                    }
                }
                
                # Add feature importance if available
                if 'feature_importance' in MODEL_INFO['post_qualifying_model']:
                    info["top_features"] = MODEL_INFO['post_qualifying_model']['feature_importance'][:10]
                    
                # Add expected feature order for debugging
                if 'feature_columns' in MODEL_INFO['pre_qualifying_model']:
                    info["expected_pre_qual_features"] = MODEL_INFO['pre_qualifying_model']['feature_columns']
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": f"Failed to get model info: {str(e)}"}), 500

@bp.route("/validate_future_features")
def validate_future_features():
    """Validate that future features are properly prepared"""
    try:
        season = request.args.get("season", type=int)
        if season is None:
            from datetime import datetime
            current_date = datetime.now()
            season = current_date.year if current_date.month > 2 else current_date.year - 1
        
        # Try to load upcoming features from data/ folder
        up_paths = [
            "data/upcoming_features.csv",
            "upcoming_features.csv"
        ]
        
        up = None
        for path in up_paths:
            try:
                up = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if up is None:
            return jsonify({"error": "upcoming_features.csv not found"}), 404
        
        sub = up.loc[up["season"] == season] if "season" in up.columns else up
        
        validation = {
            "total_rows": len(sub),
            "has_season_round": "season" in sub.columns and "round" in sub.columns,
            "unique_rounds": sorted(sub["round"].unique().tolist()) if "round" in sub.columns else [],
            "has_driver_column": "driver" in sub.columns,
            "has_circuitId_column": "circuitId" in sub.columns,
            "available_columns": list(sub.columns),
            "uses_categorical": "driver" in sub.columns and "circuitId" in sub.columns,
            "missing_pre_qual_features": [f for f in PRE_QUAL_FEATURES if f not in sub.columns] if PRE_QUAL_FEATURES else [],
            "missing_post_qual_features": [f for f in FEATURE_COLUMNS if f not in sub.columns] if FEATURE_COLUMNS else [],
            "data_structure": MODEL_INFO.get('data_structure', {}) if MODEL_INFO else {},
            "expected_pre_qual_order": PRE_QUAL_FEATURES if PRE_QUAL_FEATURES else [],
            "actual_column_order": list(sub.columns)
        }
        
        # Check for data quality issues
        issues = []
        if not validation["has_season_round"]:
            issues.append("Missing season/round columns")
        
        if not validation["has_driver_column"]:
            issues.append("Missing driver information")
        
        if len(validation["missing_pre_qual_features"]) > 5:
            issues.append(f"Too many missing pre-qualifying features: {len(validation['missing_pre_qual_features'])}")
        
        if validation["total_rows"] == 0:
            issues.append("No data for specified season")
        
        validation["issues"] = issues
        validation["ready_for_prediction"] = len(issues) == 0
        validation["recommended_approach"] = "enhanced_categorical"
        
        return jsonify(validation)
        
    except Exception as e:
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500

@bp.route("/health")
def health_check():
    """Simple health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "models_loaded": {
                "fastest_lap": fastest_model is not None,
                "finish_position": finish_model is not None,
                "pre_qualifying": finish_model_pre_qual is not None
            },
            "encoders_loaded": {
                "pre_qualifying": bool(ENCODERS_PRE),
                "post_qualifying": bool(ENCODERS_POST)
            },
            "features": {
                "categorical_support": bool(CATEGORICAL_FEATURES),
                "total_features": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0
            },
            "data_structure": MODEL_INFO.get('data_structure', {}) if MODEL_INFO else {},
            "current_season": MODEL_INFO.get('current_season') if MODEL_INFO else None
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
    
# Add this endpoint to your predictor_api.py file

@bp.route("/available_races")
def get_available_races():
    """Get available races from the predictor features CSV"""
    try:
        # Load your circuits data
        circuit_data = {
            "bahrain": { "name": "Bahrain International Circuit", "grandPrix": "Bahrain Grand Prix" },
            "jeddah": { "name": "Jeddah Corniche Circuit", "grandPrix": "Saudi Arabian Grand Prix" },
            "albert_park": { "name": "Albert Park Grand Prix Circuit", "grandPrix": "Australian Grand Prix" },
            "shanghai": { "name": "Shanghai International Circuit", "grandPrix": "Chinese Grand Prix" },
            "miami": { "name": "Miami International Autodrome", "grandPrix": "Miami Grand Prix" },
            "imola": { "name": "Autodromo Enzo e Dino Ferrari", "grandPrix": "Emilia Romagna Grand Prix" },
            "catalunya": { "name": "Circuit de Barcelona-Catalunya", "grandPrix": "Spanish Grand Prix" },
            "monaco": { "name": "Circuit de Monaco", "grandPrix": "Monaco Grand Prix" },
            "baku": { "name": "Baku City Circuit", "grandPrix": "Azerbaijan Grand Prix" },
            "villeneuve": { "name": "Circuit Gilles Villeneuve", "grandPrix": "Canadian Grand Prix" },
            "red_bull_ring": { "name": "Red Bull Ring", "grandPrix": "Austrian Grand Prix" },
            "silverstone": { "name": "Silverstone Circuit", "grandPrix": "British Grand Prix" },
            "hungaroring": { "name": "Hungaroring", "grandPrix": "Hungarian Grand Prix" },
            "spa": { "name": "Circuit de Spa-Francorchamps", "grandPrix": "Belgian Grand Prix" },
            "zandvoort": { "name": "Circuit Zandvoort", "grandPrix": "Dutch Grand Prix" },
            "monza": { "name": "Autodromo Nazionale Monza", "grandPrix": "Italian Grand Prix" },
            "marina_bay": { "name": "Marina Bay Street Circuit", "grandPrix": "Singapore Grand Prix" },
            "suzuka": { "name": "Suzuka International Racing Course", "grandPrix": "Japanese Grand Prix" },
            "lusail": { "name": "Lusail International Circuit", "grandPrix": "Qatar Grand Prix" },
            "americas": { "name": "Circuit of the Americas", "grandPrix": "United States Grand Prix" },
            "rodriguez": { "name": "Autódromo Hermanos Rodríguez", "grandPrix": "Mexican Grand Prix" },
            "interlagos": { "name": "Autódromo José Carlos Pace", "grandPrix": "Brazilian Grand Prix" },
            "las_vegas": { "name": "Las Vegas Strip Circuit", "grandPrix": "Las Vegas Grand Prix" },
            "yas_marina": { "name": "Yas Marina Circuit", "grandPrix": "Abu Dhabi Grand Prix" }
        }
        
        # Try to load data from your existing file structure
        data_paths = [
            "data/predictor_features.csv",
            "predictor_features.csv"
        ]
        
        feats = None
        for path in data_paths:
            try:
                feats = pd.read_csv(path)
                print(f"✅ Loaded race data from {path}")
                break
            except FileNotFoundError:
                continue
        
        if feats is None:
            return jsonify({"error": "predictor_features.csv not found"}), 404
        
        # Get unique combinations of season, round, and circuitId
        if 'circuitId' in feats.columns:
            race_info = feats.groupby(['season', 'round', 'circuitId']).first().reset_index()
        else:
            # Fallback if no circuitId column
            race_info = feats.groupby(['season', 'round']).first().reset_index()
            race_info['circuitId'] = 'unknown'
        
        # Organize by season
        races_by_season = {}
        available_seasons = sorted(race_info['season'].unique().tolist())
        
        for season in available_seasons:
            season_races = race_info[race_info['season'] == season].copy()
            season_races = season_races.sort_values('round')
            
            races_list = []
            for _, race in season_races.iterrows():
                circuit_id = race.get('circuitId', 'unknown')
                
                # Get race name from circuit data
                circuit_info = circuit_data.get(circuit_id, {})
                race_name = circuit_info.get('grandPrix', f'Round {race["round"]} Grand Prix')
                
                races_list.append({
                    'id': circuit_id,
                    'round': int(race['round']),
                    'name': race_name,
                    'circuit_name': circuit_info.get('name', 'Unknown Circuit'),
                    'date': race.get('race_date', 'Unknown'),
                    'circuitId': circuit_id
                })
            
            races_by_season[str(season)] = races_list
        
        return jsonify({
            'races': races_by_season,
            'seasons': available_seasons,
            'total_races': len(race_info),
            'has_circuit_data': 'circuitId' in feats.columns,
            'data_source': 'predictor_features.csv'
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to load available races: {str(e)}"}), 500

# Initialize models on import
try:
    load_models()
    print("✅ Enhanced predictor API loaded with proper feature handling")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    print("Run feature engineering and training scripts first")

if __name__ == "__main__":
    print("Enhanced F1 Predictor API ready")
    print("Features:")
    print("- Proper categorical encoding")
    print("- Correct feature ordering")
    print("- Enhanced error handling")
    print("- Data-driven approach maintained")