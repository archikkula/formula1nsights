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
globalfinish_model = None
finish_model_pre_qual = None
grid_model = None
FEATURE_COLUMNS = None
PRE_QUAL_FEATURES = None
CATEGORICAL_FEATURES = None
ENCODERS_PRE = None
ENCODERS_POST = None
MODEL_INFO = None
GRID_ENCODERS = None
GRID_FEATURES = None

def load_models():
    
    """Load models and encoders with data/ folder structure support"""
    global finish_model, finish_model_pre_qual
    global FEATURE_COLUMNS, PRE_QUAL_FEATURES, CATEGORICAL_FEATURES
    global ENCODERS_PRE, ENCODERS_POST, MODEL_INFO
    global grid_model, GRID_ENCODERS, GRID_FEATURES
    
    try:
        
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
        elif 'grid_position_model' in MODEL_INFO:
            # We have grid model but missing race model info - use fallback
            print("⚠️  Race model features not in model_info.json, using fallback")
            FEATURE_COLUMNS = [
                'leak_proof_team_mean', 'corners', 'driver', 'leak_proof_points_rate', 
                'length_km', 'leak_proof_peak_performance', 'leak_proof_current_podium_rate', 
                'leak_proof_circuit_podium_norm', 'Q1', 'leak_proof_last3_fin', 'grid_position', 
                'leak_proof_weighted_podium_rate', 'team', 'leak_proof_career_mean', 'precip_mm', 
                'leak_proof_recent_form', 'leak_proof_weighted_mean', 'leak_proof_circuit_mean_norm', 
                'leak_proof_current_wins_rate', 'made_Q2', 'made_Q3', 'avg_temp', 'circuitId', 
                'laps', 'leak_proof_consistency', 'Q2_Q3_ratio', 'Q3', 'leak_proof_career_podium', 
                'leak_proof_current_mean', 'Q2', 'leak_proof_team_podium_rate', 'Q1_Q2_ratio'
            ]
            PRE_QUAL_FEATURES = [f for f in FEATURE_COLUMNS if f not in ['Q1', 'Q2', 'Q3', 'Q1_Q2_ratio', 'Q2_Q3_ratio', 'made_Q2', 'made_Q3', 'grid_position']]
            CATEGORICAL_FEATURES = ['driver', 'circuitId', 'team']
        else:
            # Old format - backward compatibility
            FEATURE_COLUMNS = MODEL_INFO.get('feature_columns', [])
            PRE_QUAL_FEATURES = FEATURE_COLUMNS
            CATEGORICAL_FEATURES = MODEL_INFO.get('categorical_features', [])
        
        # Load grid position model
        try:
            grid_model = joblib.load("models/xgb_grid_predictor.pkl")
            print("✅ Loaded grid position predictor")
        except FileNotFoundError:
            print("⚠️  Grid position predictor not found")
            grid_model = None
        
        try:
            GRID_ENCODERS = joblib.load("models/label_encoders_grid.pkl")
            print("✅ Loaded grid position encoders")
        except FileNotFoundError:
            print("⚠️  Grid position encoders not found")
            GRID_ENCODERS = {}
        
        # Get grid features from model info
        if MODEL_INFO and 'grid_position_model' in MODEL_INFO:
            GRID_FEATURES = MODEL_INFO['grid_position_model'].get('feature_columns', [])
            print(f"✅ Grid model features: {len(GRID_FEATURES)}")
        else:
            GRID_FEATURES = []
        
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


@bp.route("/model_info")
def get_model_info():
    """Get information about loaded models and their performance"""
    try:
        info = {
            "models_loaded": {
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
                "finish_position": finish_model is not None,
                "pre_qualifying": finish_model_pre_qual is not None,
                "grid_position": grid_model is not None
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
    
@bp.route("/predict_grid_positions")
def predict_grid_positions():
    """Predict qualifying grid positions for a race"""
    try:
        season = request.args.get("season", type=int)
        round_num = request.args.get("round", type=int)
        include_confidence = request.args.get("confidence", "false").lower() == "true"
        
        if grid_model is None:
            return jsonify({"error": "Grid position predictor not loaded"}), 404
        
        # Try to load upcoming features
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
        
        # Filter data
        sub = up.copy()
        if season is not None:
            sub = sub[sub["season"] == season] if "season" in sub.columns else sub
        if round_num is not None:
            sub = sub[sub["round"] == round_num] if "round" in sub.columns else sub
        
        if sub.empty:
            return jsonify({"error": f"No data for season {season}, round {round_num}"}), 404
        
        # Prepare features for grid prediction using the grid-specific encoders
        X = prepare_features_enhanced(sub, GRID_FEATURES, GRID_ENCODERS, ['driver', 'circuitId', 'team'], "grid_prediction")
        
        if include_confidence:
            grid_scores, confidence = get_prediction_confidence(grid_model, X)
        else:
            grid_scores = grid_model.predict(X)
            confidence = None
        
        # Convert scores to grid positions (rank them)
        order = np.argsort(grid_scores)
        predicted_grid = np.empty_like(order, dtype=float)
        predicted_grid[order] = np.arange(1, len(order) + 1)
        
        drivers = sub["driver"].tolist() if "driver" in sub.columns else ["unknown"] * len(sub)
        
        results = []
        for i, (driver, pos, score) in enumerate(zip(drivers, predicted_grid, grid_scores)):
            entry = {
                "driver": driver,
                "predicted_grid": float(pos),
                "model_score": float(score)
            }
            if confidence is not None:
                entry["confidence"] = float(confidence[i])
            results.append(entry)
        
        # Sort by predicted grid position
        results.sort(key=lambda x: x["predicted_grid"])
        
        return jsonify({
            "season": season,
            "round": round_num,
            "grid_predictions": results,
            "total_drivers": len(results),
            "model_type": "grid_position"
        })
        
    except Exception as e:
        return jsonify({"error": f"Grid prediction failed: {str(e)}"}), 500

@bp.route("/predict_race_with_predicted_grid")
def predict_race_with_predicted_grid():
    """Predict race results using predicted grid positions instead of actual ones"""
    try:
        season = request.args.get("season", type=int)
        round_num = request.args.get("round", type=int)
        include_confidence = request.args.get("confidence", "false").lower() == "true"
        next_only = request.args.get("next_only", "false").lower() == "true"  # NEW PARAMETER
        
        if grid_model is None or finish_model is None:
            return jsonify({"error": "Required models not loaded"}), 404
        
        # Load upcoming features
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
        
        # Filter data
        sub = up.copy()
        if season is not None:
            sub = sub[sub["season"] == season] if "season" in sub.columns else sub
        
        if sub.empty:
            return jsonify({"error": f"No data for season {season}"}), 404
        
        # NEW LOGIC: Handle next_only parameter
        if next_only:
            # Get only the next race (minimum round number)
            next_round = sub["round"].min() if "round" in sub.columns else 1
            sub = sub[sub["round"] == next_round]
        elif round_num is not None:
            # Specific round requested
            sub = sub[sub["round"] == round_num] if "round" in sub.columns else sub
        # If neither next_only nor round_num specified, return all rounds (existing behavior)
        
        if sub.empty:
            return jsonify({"error": f"No data for specified parameters"}), 404
        
        # Step 1: Predict grid positions
        X_grid = prepare_features_enhanced(sub, GRID_FEATURES, GRID_ENCODERS, ['driver', 'circuitId', 'team'], "grid_prediction")
        grid_scores = grid_model.predict(X_grid)
        
        # Convert to grid positions
        order = np.argsort(grid_scores)
        predicted_grid = np.empty_like(order, dtype=float)
        predicted_grid[order] = np.arange(1, len(order) + 1)
        
        # Step 2: Use predicted grid positions for race prediction
        sub_with_predicted_grid = sub.copy()
        sub_with_predicted_grid['grid_position'] = predicted_grid
        
        # Create mock qualifying data based on grid position
        sub_with_predicted_grid['made_Q3'] = (predicted_grid <= 10).astype(int)
        sub_with_predicted_grid['made_Q2'] = (predicted_grid <= 15).astype(int)
        
        # Create mock qualifying times based on grid position (for ratio calculations)
        base_time = 90.0
        sub_with_predicted_grid['Q1'] = base_time + predicted_grid * 0.5
        sub_with_predicted_grid['Q2'] = np.where(
            predicted_grid <= 15, 
            base_time + predicted_grid * 0.4, 
            np.nan
        )
        sub_with_predicted_grid['Q3'] = np.where(
            predicted_grid <= 10, 
            base_time + predicted_grid * 0.3, 
            np.nan
        )
        
        # Calculate Q1_Q2_ratio and Q2_Q3_ratio if they're expected features
        if 'Q1_Q2_ratio' in FEATURE_COLUMNS:
            sub_with_predicted_grid['Q1_Q2_ratio'] = np.where(
                sub_with_predicted_grid['Q2'].notna(),
                sub_with_predicted_grid['Q1'] / sub_with_predicted_grid['Q2'],
                1.0
            )
        
        if 'Q2_Q3_ratio' in FEATURE_COLUMNS:
            sub_with_predicted_grid['Q2_Q3_ratio'] = np.where(
                sub_with_predicted_grid['Q3'].notna(),
                sub_with_predicted_grid['Q2'] / sub_with_predicted_grid['Q3'],
                1.0
            )
        
        # Prepare features for race prediction
        X_race = prepare_features_enhanced(
            sub_with_predicted_grid, 
            FEATURE_COLUMNS, 
            ENCODERS_POST, 
            CATEGORICAL_FEATURES, 
            "post_qualifying"
        )
        
        if include_confidence:
            race_scores, race_confidence = get_prediction_confidence(finish_model, X_race)
        else:
            race_scores = finish_model.predict(X_race)
            race_confidence = None
        
        # Convert to race positions
        race_order = np.argsort(race_scores)
        predicted_race = np.empty_like(race_order, dtype=float)
        predicted_race[race_order] = np.arange(1, len(race_order) + 1)
        
        drivers = sub["driver"].tolist() if "driver" in sub.columns else ["unknown"] * len(sub)
        
        results = []
        for i, driver in enumerate(drivers):
            entry = {
                "driver": driver,
                "predicted_grid": float(predicted_grid[i]),
                "predicted_finish": float(predicted_race[i]),
                "grid_model_score": float(grid_scores[i]),
                "race_model_score": float(race_scores[i])
            }
            
            # Add round info if available
            if "round" in sub.columns:
                entry["round"] = int(sub.iloc[i]["round"])
            
            if race_confidence is not None:
                entry["race_confidence"] = float(race_confidence[i])
            results.append(entry)
        
        # Sort by predicted finish position
        results.sort(key=lambda x: x["predicted_finish"])
        
        response_data = {
            "season": season,
            "predictions": results,
            "total_drivers": len(results),
            "model_type": "two_stage_prediction"
        }
        
        # Add round info if filtering by specific round or next_only
        if next_only and "round" in sub.columns:
            response_data["round"] = int(sub["round"].iloc[0])
            response_data["description"] = "Next race only"
        elif round_num is not None:
            response_data["round"] = round_num
            response_data["description"] = f"Round {round_num} specific"
        else:
            response_data["description"] = "All available rounds"
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"Two-stage prediction failed: {str(e)}"}), 500

@bp.route("/compare_grid_predictions")
def compare_grid_predictions():
    """Compare predicted vs actual grid positions for completed races"""
    try:
        season = request.args.get("season", type=int)
        round_num = request.args.get("round", type=int)
        
        if grid_model is None:
            return jsonify({"error": "Grid position predictor not loaded"}), 404
        
        # Load historical features (has actual grid positions)
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
            return jsonify({"error": "predictor_features.csv not found"}), 404
        
        # Filter for specific race
        sub = feats.loc[(feats["season"] == season) & (feats["round"] == round_num)]
        
        if sub.empty:
            return jsonify({"error": f"No data for season {season}, round {round_num}"}), 404
        
        # Prepare features for grid prediction (exclude actual grid data)
        X_grid = prepare_features_enhanced(sub, GRID_FEATURES, GRID_ENCODERS, ['driver', 'circuitId', 'team'], "grid_prediction")
        predicted_grid_scores = grid_model.predict(X_grid)
        
        # Convert to grid positions
        order = np.argsort(predicted_grid_scores)
        predicted_grid = np.empty_like(order, dtype=float)
        predicted_grid[order] = np.arange(1, len(order) + 1)
        
        drivers = sub["driver"].tolist() if "driver" in sub.columns else ["unknown"] * len(sub)
        actual_grid = sub["grid_position"].tolist() if "grid_position" in sub.columns else [None] * len(sub)
        
        results = []
        for i, driver in enumerate(drivers):
            entry = {
                "driver": driver,
                "predicted_grid": float(predicted_grid[i]),
                "actual_grid": float(actual_grid[i]) if actual_grid[i] is not None else None,
                "grid_difference": float(predicted_grid[i] - actual_grid[i]) if actual_grid[i] is not None else None,
                "model_score": float(predicted_grid_scores[i])
            }
            results.append(entry)
        
        # Sort by actual grid position
        results.sort(key=lambda x: x["actual_grid"] if x["actual_grid"] is not None else 999)
        
        # Calculate accuracy metrics
        if all(r["actual_grid"] is not None for r in results):
            predicted = [r["predicted_grid"] for r in results]
            actual = [r["actual_grid"] for r in results]
            
            mae = np.mean([abs(p - a) for p, a in zip(predicted, actual)])
            rmse = np.sqrt(np.mean([(p - a)**2 for p, a in zip(predicted, actual)]))
            
            # Spearman correlation
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(predicted, actual)
        else:
            mae = rmse = spearman_corr = None
        
        return jsonify({
            "season": season,
            "round": round_num,
            "grid_comparison": results,
            "total_drivers": len(results),
            "accuracy_metrics": {
                "mae": mae,
                "rmse": rmse,
                "spearman_correlation": spearman_corr
            },
            "model_type": "grid_position_comparison"
        })
        
    except Exception as e:
        return jsonify({"error": f"Grid comparison failed: {str(e)}"}), 500
    
@bp.route("/predict_historical_race_two_stage")
def predict_historical_race_two_stage():
    """Predict historical race using two-stage model - TRULY LEAK-FREE"""
    try:
        season = request.args.get("season", type=int)
        round_num = request.args.get("round", type=int)
        include_confidence = request.args.get("confidence", "false").lower() == "true"
        
        if grid_model is None or finish_model is None:
            return jsonify({"error": "Required models not loaded"}), 404
        
        # Load historical features
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
            return jsonify({"error": "predictor_features.csv not found"}), 404
        
        # Get the target race
        target_race = feats[(feats["season"] == season) & (feats["round"] == round_num)].copy()
        
        if target_race.empty:
            return jsonify({"error": f"No data for season {season}, round {round_num}"}), 404
        
        # CRITICAL: Use chronological filtering - only races that happened BEFORE this race
        # Filter by (season, round) combination to ensure chronological order
        historical_data = feats[
            (feats['season'] < season) | 
            ((feats['season'] == season) & (feats['round'] < round_num))
        ].copy()
        
        print(f"🔍 Target race: Season {season}, Round {round_num}")
        print(f"📊 Historical data: {len(historical_data)} races before this race")
        
        if historical_data.empty:
            return jsonify({
                "error": "No historical data available before this race",
                "suggestion": "Try a later race in the season or a later season"
            }), 404
        
        # Calculate truly leak-free features for each driver
        clean_features = []
        
        for _, driver_row in target_race.iterrows():
            driver = driver_row['driver']
            circuit_id = driver_row['circuitId']
            team = driver_row['team']
            
            # Get driver's complete history BEFORE this race
            driver_history = historical_data[historical_data['driver'] == driver].copy()
            team_history = historical_data[historical_data['team'] == team].copy()
            circuit_history = historical_data[historical_data['circuitId'] == circuit_id].copy()
            
            # Driver's circuit-specific history
            driver_circuit_history = driver_history[driver_history['circuitId'] == circuit_id].copy()
            
            # Current season history (races before this round only)
            current_season_history = driver_history[driver_history['season'] == season].copy()
            
            print(f"👤 {driver}: {len(driver_history)} total races, {len(current_season_history)} this season, {len(driver_circuit_history)} at this circuit")
            
            # Calculate leak-free features using only historical data
            clean_row = {
                'driver': driver,
                'team': team,
                'circuitId': circuit_id,
                'season': season,
                'round': round_num,
                
                # Static circuit characteristics (no leak possible)
                'length_km': driver_row['length_km'],
                'corners': driver_row['corners'],
                'laps': driver_row['laps'],
                'avg_temp': driver_row['avg_temp'],
                'precip_mm': driver_row['precip_mm'],
                
                # Career statistics (before this race)
                'leak_proof_career_mean': driver_history['finish_pos'].mean() if len(driver_history) > 0 else 11.0,
                'leak_proof_career_podium': (driver_history['finish_pos'] <= 3).mean() if len(driver_history) > 0 else 0.05,
                
                # Recent form (last few races)
                'leak_proof_last3_fin': driver_history.tail(3)['finish_pos'].mean() if len(driver_history) >= 3 else 11.0,
                'leak_proof_recent_form': driver_history.tail(5)['finish_pos'].mean() if len(driver_history) >= 5 else 11.0,
                
                # Performance consistency
                'leak_proof_consistency': driver_history['finish_pos'].std() if len(driver_history) > 1 else 6.0,
                'leak_proof_peak_performance': driver_history['finish_pos'].min() if len(driver_history) > 0 else 20.0,
                
                # Points scoring rate
                'leak_proof_points_rate': (driver_history['finish_pos'] <= 10).mean() if len(driver_history) > 0 else 0.3,
                
                # Team performance (before this race)
                'leak_proof_team_mean': team_history['finish_pos'].mean() if len(team_history) > 0 else 11.0,
                'leak_proof_team_podium_rate': (team_history['finish_pos'] <= 3).mean() if len(team_history) > 0 else 0.1,
                
                # Current season performance (races completed before this one)
                'leak_proof_current_mean': current_season_history['finish_pos'].mean() if len(current_season_history) > 0 else 11.0,
                'leak_proof_current_podium_rate': (current_season_history['finish_pos'] <= 3).mean() if len(current_season_history) > 0 else 0.05,
                'leak_proof_current_wins_rate': (current_season_history['finish_pos'] == 1).mean() if len(current_season_history) > 0 else 0.02,
                
                # Weighted recent performance (more weight to recent races)
                'leak_proof_weighted_mean': 11.0,
                'leak_proof_weighted_podium_rate': 0.05,
                
                # Circuit-specific performance
                'leak_proof_circuit_mean_norm': 0.0,
                'leak_proof_circuit_podium_norm': 0.0
            }
            
            # Calculate weighted recent performance (last 10 races with exponential decay)
            if len(driver_history) > 0:
                recent_races = driver_history.tail(10).copy()
                if len(recent_races) > 0:
                    # Create exponential weights (more recent = higher weight)
                    weights = np.exp(np.linspace(-1, 0, len(recent_races)))
                    weights = weights / weights.sum()
                    
                    clean_row['leak_proof_weighted_mean'] = np.average(recent_races['finish_pos'], weights=weights)
                    clean_row['leak_proof_weighted_podium_rate'] = np.average((recent_races['finish_pos'] <= 3).astype(float), weights=weights)
            
            # Circuit-specific performance normalization
            if len(driver_circuit_history) > 0 and len(circuit_history) > 0:
                driver_circuit_avg = driver_circuit_history['finish_pos'].mean()
                circuit_avg = circuit_history['finish_pos'].mean()
                
                # Normalize driver's circuit performance relative to overall circuit difficulty
                clean_row['leak_proof_circuit_mean_norm'] = driver_circuit_avg - circuit_avg
                clean_row['leak_proof_circuit_podium_norm'] = (driver_circuit_history['finish_pos'] <= 3).mean()
            
            clean_features.append(clean_row)
        
        # Convert to DataFrame and remove qualifying data
        sub_clean = pd.DataFrame(clean_features)
        
        print(f"✅ Created leak-free features for {len(sub_clean)} drivers")
        
        # Step 1: Predict grid positions using clean features
        X_grid = prepare_features_enhanced(
            sub_clean, 
            GRID_FEATURES, 
            GRID_ENCODERS, 
            ['driver', 'circuitId', 'team'], 
            "grid_prediction"
        )
        grid_scores = grid_model.predict(X_grid)
        
        # Convert to grid positions
        order = np.argsort(grid_scores)
        predicted_grid = np.empty_like(order, dtype=float)
        predicted_grid[order] = np.arange(1, len(order) + 1)
        
        # Step 2: Use predicted grid for race prediction
        sub_with_predicted_grid = sub_clean.copy()
        sub_with_predicted_grid['grid_position'] = predicted_grid
        sub_with_predicted_grid['made_Q3'] = (predicted_grid <= 10).astype(int)
        sub_with_predicted_grid['made_Q2'] = (predicted_grid <= 15).astype(int)
        
        # Create mock qualifying times based on predicted grid
        base_time = 90.0
        sub_with_predicted_grid['Q1'] = base_time + predicted_grid * 0.5
        sub_with_predicted_grid['Q2'] = np.where(
            predicted_grid <= 15, 
            base_time + predicted_grid * 0.4, 
            np.nan
        )
        sub_with_predicted_grid['Q3'] = np.where(
            predicted_grid <= 10, 
            base_time + predicted_grid * 0.3, 
            np.nan
        )
        
        # Calculate timing ratios if needed
        if 'Q1_Q2_ratio' in FEATURE_COLUMNS:
            sub_with_predicted_grid['Q1_Q2_ratio'] = np.where(
                sub_with_predicted_grid['Q2'].notna(),
                sub_with_predicted_grid['Q1'] / sub_with_predicted_grid['Q2'],
                1.0
            )
        
        if 'Q2_Q3_ratio' in FEATURE_COLUMNS:
            sub_with_predicted_grid['Q2_Q3_ratio'] = np.where(
                sub_with_predicted_grid['Q3'].notna(),
                sub_with_predicted_grid['Q2'] / sub_with_predicted_grid['Q3'],
                1.0
            )
        
        # Step 3: Predict race results using clean features + predicted grid
        X_race = prepare_features_enhanced(
            sub_with_predicted_grid, 
            FEATURE_COLUMNS, 
            ENCODERS_POST, 
            CATEGORICAL_FEATURES, 
            "post_qualifying"
        )
        
        if include_confidence:
            race_scores, race_confidence = get_prediction_confidence(finish_model, X_race)
        else:
            race_scores = finish_model.predict(X_race)
            race_confidence = None
        
        # Convert to race positions
        race_order = np.argsort(race_scores)
        predicted_race = np.empty_like(race_order, dtype=float)
        predicted_race[race_order] = np.arange(1, len(race_order) + 1)
        
        # Get actual results for comparison
        actual_finish = target_race["finish_pos"].tolist()
        actual_grid = target_race["grid_position"].tolist()
        
        results = []
        for i, driver in enumerate(sub_clean['driver']):
            # Get historical context for this driver
            driver_hist_count = len(historical_data[historical_data['driver'] == driver])
            
            entry = {
                "driver": driver,
                "predicted_grid": float(predicted_grid[i]),
                "predicted_finish": float(predicted_race[i]),
                "actual_grid": float(actual_grid[i]) if actual_grid[i] is not None else None,
                "actual_finish": float(actual_finish[i]) if actual_finish[i] is not None else None,
                "grid_model_score": float(grid_scores[i]),
                "race_model_score": float(race_scores[i]),
                "historical_races": driver_hist_count  # Shows how much data we had for prediction
            }
            if race_confidence is not None:
                entry["race_confidence"] = float(race_confidence[i])
            results.append(entry)
        
        # Sort by predicted finish position
        results.sort(key=lambda x: x["predicted_finish"])
        
        return jsonify({
            "season": season,
            "round": round_num,
            "predictions": results,
            "total_drivers": len(results),
            "model_type": "historical_two_stage_leak_free",
            "description": "Truly leak-free historical prediction using only pre-race data",
            "historical_races_used": len(historical_data),
            "data_integrity": "leak_free"
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Leak-free historical prediction error: {error_details}")
        return jsonify({"error": f"Leak-free prediction failed: {str(e)}"}), 500
    
#initialize models
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