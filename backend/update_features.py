import pandas as pd
import numpy as np

def analyze_2025_performance():
    df = pd.read_csv("data/race_history.csv")
    
    df_2025 = df[df['season'] == 2025].copy()

    driver_stats = []
    
    for driver in df_2025['driver'].unique():
        driver_data = df_2025[df_2025['driver'] == driver]
        
        if len(driver_data) > 0:
            avg_finish = driver_data['finish_pos'].mean()
            podiums = len(driver_data[driver_data['finish_pos'] <= 3])
            podium_rate = podiums / len(driver_data)
            races = len(driver_data)
            
            # Last 3 races performance
            last_3 = driver_data.nlargest(3, 'round')['finish_pos'].values
            last_3_avg = last_3.mean() if len(last_3) > 0 else None
            
            driver_stats.append({
                'driver': driver,
                'races': races,
                'avg_finish': avg_finish,
                'podiums': podiums,
                'podium_rate': podium_rate,
                'last_3_avg': last_3_avg,
                'last_3_results': list(last_3) if len(last_3) > 0 else []
            })
    
    # Sort by average finish (best first)
    driver_stats.sort(key=lambda x: x['avg_finish'])
    
    print("Current 2025 Championship Order:")
    print("-" * 60)
    for i, stats in enumerate(driver_stats[:15], 1):
        print(f"{i:2d}. {stats['driver']:15s} | Avg: {stats['avg_finish']:4.1f} | "
              f"Podiums: {stats['podiums']:2d}/{stats['races']:2d} | "
              f"Rate: {stats['podium_rate']:4.2f} | Last 3: {stats['last_3_results']}")
    
    return driver_stats

def check_feature_accuracy():
   
    features = pd.read_csv("data/predictor_features.csv")
    round1_features = features[(features['season'] == 2025) & (features['round'] == 1)]
    
    history = pd.read_csv("data/race_history.csv")
    
    print("Checking if features match actual 2025 performance...")
    
    key_drivers = ['norris', 'piastri', 'hamilton', 'leclerc', 'max_verstappen', 'russell']
    
    for driver in key_drivers:
        if f'driver_{driver}' in round1_features.columns:
            
            driver_mask = round1_features[f'driver_{driver}'] == 1
            if driver_mask.any():
                driver_row = round1_features[driver_mask].iloc[0]
                
                driver_history = history[
                    (history['driver'] == driver) & 
                    ((history['season'] < 2025) | 
                     ((history['season'] == 2025) & (history['round'] < 1)))
                ]
                
                if len(driver_history) > 0:
                    actual_career_avg = driver_history['finish_pos'].mean()
                    actual_podiums = len(driver_history[driver_history['finish_pos'] <= 3])
                    actual_podium_rate = actual_podiums / len(driver_history)
                    
                    last_3_races = driver_history.nlargest(3, ['season', 'round'])
                    actual_last3_avg = last_3_races['finish_pos'].mean() if len(last_3_races) > 0 else None
                    
                    print(f"\n{driver.upper()}:")
                    print(f"  Feature career_mean: {driver_row['career_mean']:.2f} | Actual: {actual_career_avg:.2f}")
                    print(f"  Feature podium_rate: {driver_row['podium_rate']:.3f} | Actual: {actual_podium_rate:.3f}")
                    if actual_last3_avg:
                        print(f"  Feature last3_fin: {driver_row['last3_fin']:.2f} | Actual: {actual_last3_avg:.2f}")

def update_features_with_2025_data():
   
    features = pd.read_csv("data/predictor_features.csv")
    history = pd.read_csv("data/race_history.csv")
    
   
    
    updated_features = features.copy()
    
    for idx, row in features.iterrows():
        season = row['season']
        round_num = row['round']
       
        driver_cols = [col for col in features.columns if col.startswith('driver_')]
        driver = None
        for col in driver_cols:
            if row[col] == 1:
                driver = col.replace('driver_', '')
                break
        
        if driver:
          
            driver_history = history[
                (history['driver'] == driver) & 
                ((history['season'] < season) | 
                 ((history['season'] == season) & (history['round'] < round_num)))
            ]
            
            if len(driver_history) > 0:
                # Recalculate career stats
                updated_features.at[idx, 'career_mean'] = driver_history['finish_pos'].mean()
                
                podiums = len(driver_history[driver_history['finish_pos'] <= 3])
                updated_features.at[idx, 'career_podium'] = podiums
                updated_features.at[idx, 'podium_rate'] = podiums / len(driver_history)
                
                # Last 3 races
                last_3_races = driver_history.nlargest(3, ['season', 'round'])
                if len(last_3_races) > 0:
                    updated_features.at[idx, 'last3_fin'] = last_3_races['finish_pos'].mean()
                    updated_features.at[idx, 'last3_lap'] = last_3_races['fastest_lap'].mean()
    
    # Save updated features
    updated_features.to_csv("data/predictor_features_updated.csv", index=False)
    print("Updated features saved to data/predictor_features_updated.csv")
    
    return updated_features

def compare_predictions():
    """Compare predictions with old vs new features"""
    print("\n=== COMPARING OLD VS NEW PREDICTIONS ===\n")
    
    try:
        import joblib
        
        # Load model
        model = joblib.load("models/xgb_finish_ranker.pkl")
        
        # Load feature info
        with open('data/model_info.json', 'r') as f:
            model_info = json.load(f)
        feature_columns = model_info['feature_columns']
        
        # Test on 2025 Round 1
        old_features = pd.read_csv("data/predictor_features.csv")
        new_features = pd.read_csv("data/predictor_features_updated.csv")
        
        old_round1 = old_features[(old_features['season'] == 2025) & (old_features['round'] == 1)]
        new_round1 = new_features[(new_features['season'] == 2025) & (new_features['round'] == 1)]
        
        if len(old_round1) > 0 and len(new_round1) > 0:
            # Prepare features
            def prepare_features(df):
                X = pd.DataFrame()
                for col in feature_columns:
                    if col in df.columns:
                        X[col] = df[col]
                    else:
                        X[col] = 0.0
                return X.fillna(0.0)
            
            old_X = prepare_features(old_round1)
            new_X = prepare_features(new_round1)
            
            # Make predictions
            old_pred = model.predict(old_X)
            new_pred = model.predict(new_X)
            
            # Convert to rankings
            old_ranks = np.argsort(old_pred) + 1
            new_ranks = np.argsort(new_pred) + 1
            
            # Get driver names
            driver_cols = [col for col in old_round1.columns if col.startswith('driver_')]
            drivers = []
            for _, row in old_round1.iterrows():
                for col in driver_cols:
                    if row[col] == 1:
                        drivers.append(col.replace('driver_', ''))
                        break
            
            print("Prediction Comparison for 2025 Round 1:")
            print("-" * 50)
            print(f"{'Driver':<15} {'Old Rank':<10} {'New Rank':<10} {'Change'}")
            print("-" * 50)
            
            for driver, old_rank, new_rank in zip(drivers, old_ranks, new_ranks):
                change = new_rank - old_rank
                change_str = f"{change:+d}" if change != 0 else "0"
                print(f"{driver:<15} {old_rank:<10} {new_rank:<10} {change_str}")
                
    except Exception as e:
        print(f"Could not compare predictions: {e}")

if __name__ == "__main__":
    import json
    
    # Analyze current 2025 performance
    driver_stats = analyze_2025_performance()
    
    # Check feature accuracy
    check_feature_accuracy()
    
    # Update features
    updated_features = update_features_with_2025_data()
    
    # Compare predictions
    compare_predictions()
    
    print(f"\n=== NEXT STEPS ===")
    print(f"1. Review the 2025 performance analysis above")
    print(f"2. If the updated predictions look better, replace your original file:")
    print(f"   mv data/predictor_features_updated.csv data/predictor_features.csv")
    print(f"3. Retrain your model with: python train_rank_model.py")
    print(f"4. Test the API again")