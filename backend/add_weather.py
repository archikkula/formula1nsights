import pandas as pd
import numpy as np

def check_available_weather_data():
    """Check what weather data is available"""
    df = pd.read_csv("data/predictor_features.csv")
    
    # Check for weather-related columns
    weather_cols = []
    potential_weather = ['temp', 'precip', 'rain', 'weather', 'wind', 'humid', 'pressure', 'cloud']
    
    for col in df.columns:
        if any(weather_term in col.lower() for weather_term in potential_weather):
            weather_cols.append(col)
    
    print("Available weather data:")
    for col in weather_cols:
        non_null = df[col].count()
        print(f"  {col}: {non_null}/{len(df)} races ({non_null/len(df)*100:.1f}%)")
        if non_null > 0:
            print(f"    Range: {df[col].min():.1f} to {df[col].max():.1f}")
    
    return weather_cols

def create_enhanced_weather_features(df):
    """Create weather features that matter for F1"""
    enhanced_df = df.copy()
    
    # Rain probability (most important for F1)
    if 'precip_mm' in df.columns:
        enhanced_df['is_wet_race'] = (df['precip_mm'] > 1.0).astype(int)
        enhanced_df['light_rain'] = ((df['precip_mm'] > 0.1) & (df['precip_mm'] <= 1.0)).astype(int)
        enhanced_df['heavy_rain'] = (df['precip_mm'] > 5.0).astype(int)
    
    # Temperature effects
    if 'avg_temp' in df.columns:
        enhanced_df['very_hot'] = (df['avg_temp'] > 30).astype(int)
        enhanced_df['very_cold'] = (df['avg_temp'] < 15).astype(int)
        
        # Track temperature affects tire performance
        enhanced_df['tire_degradation_risk'] = (df['avg_temp'] > 25).astype(int)
    
    # Circuit-specific weather interactions
    circuit_cols = [col for col in df.columns if col.startswith('circuitId_')]
    
    if 'precip_mm' in df.columns:
        # Wet weather specialists circuits
        wet_circuits = ['silverstone', 'spa', 'hungaroring', 'interlagos']
        for circuit in wet_circuits:
            circuit_col = f'circuitId_{circuit}'
            if circuit_col in df.columns:
                enhanced_df[f'wet_{circuit}'] = (df[circuit_col] * enhanced_df['is_wet_race']).astype(int)
    
    return enhanced_df

def add_driver_weather_performance(df):
    """Add driver-specific weather performance"""
    enhanced_df = df.copy()
    
    # Wet weather specialists (based on F1 history)
    wet_specialists = {
        'driver_hamilton': 'wet_specialist_hamilton',
        'driver_max_verstappen': 'wet_specialist_verstappen', 
        'driver_russell': 'wet_specialist_russell',
        'driver_alonso': 'wet_specialist_alonso'
    }
    
    if 'is_wet_race' in enhanced_df.columns:
        for driver_col, specialist_col in wet_specialists.items():
            if driver_col in df.columns:
                enhanced_df[specialist_col] = (df[driver_col] * enhanced_df['is_wet_race']).astype(int)
    
    return enhanced_df

def enhanced_feature_engineering_with_weather():
    """Enhanced feature engineering including weather"""
    
    # Check available weather data
    weather_cols = check_available_weather_data()
    
    # Load data
    df = pd.read_csv("data/predictor_features.csv")
    
    # Add grid position
    df['grid_position'] = np.nan
    for _, group in df.groupby(['season', 'round']):
        if not group['Q3'].isna().all():
            grid_positions = group['Q3'].rank(method='min', na_option='bottom')
            df.loc[group.index, 'grid_position'] = grid_positions
    df['grid_position'] = df['grid_position'].fillna(df['grid_position'].median())
    
    # Add weather features
    df = create_enhanced_weather_features(df)
    df = add_driver_weather_performance(df)
    
    # Core features
    core_features = [
        'grid_position',
        'made_Q2', 'made_Q3', 'Q3',
        'podium_rate', 'career_mean', 'career_podium',
        'length_km', 'corners', 'laps'
    ]
    
    # Basic weather
    weather_features = ['avg_temp', 'precip_mm']
    
    # Enhanced weather features
    enhanced_weather = [
        'is_wet_race', 'light_rain', 'heavy_rain',
        'very_hot', 'very_cold', 'tire_degradation_risk'
    ]
    
    # Key drivers
    driver_features = [
        'driver_max_verstappen', 'driver_norris', 'driver_piastri',
        'driver_leclerc', 'driver_hamilton', 'driver_russell'
    ]
    
    # Weather specialists
    weather_specialist_features = [
        'wet_specialist_hamilton', 'wet_specialist_verstappen',
        'wet_specialist_russell', 'wet_specialist_alonso'
    ]
    
    # Key circuits
    circuit_features = [
        'circuitId_monaco', 'circuitId_silverstone', 'circuitId_spa',
        'circuitId_monza', 'circuitId_hungaroring'
    ]
    
    # Wet circuit interactions
    wet_circuit_features = [
        'wet_silverstone', 'wet_spa', 'wet_hungaroring', 'wet_interlagos'
    ]
    
    # Combine all features
    all_features = (core_features + weather_features + enhanced_weather + 
                   driver_features + weather_specialist_features + 
                   circuit_features + wet_circuit_features)
    
    # Keep only available features
    available_features = [f for f in all_features if f in df.columns]
    
    # Clean data
    for col in available_features:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(False)
    
    # Save enhanced data
    df.to_csv("data/predictor_features_enhanced.csv", index=False)
    
    # Save feature info
    feature_info = {
        'feature_list': available_features,
        'n_features': len(available_features),
        'weather_features': enhanced_weather,
        'weather_specialists': weather_specialist_features
    }
    
    import json
    with open('data/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"Enhanced features with weather: {len(available_features)}")
    print(f"Weather features added: {len([f for f in enhanced_weather if f in available_features])}")
    print(f"Weather specialists: {len([f for f in weather_specialist_features if f in available_features])}")
    
    return df, available_features

def analyze_weather_impact():
    """Analyze how weather affects F1 results"""
    df = pd.read_csv("data/predictor_features.csv")
    
    if 'precip_mm' not in df.columns:
        print("No precipitation data available")
        return
    
    # Analyze wet vs dry races
    wet_races = df[df['precip_mm'] > 0.1]
    dry_races = df[df['precip_mm'] <= 0.1]
    
    print(f"Wet races: {len(wet_races)} ({len(wet_races)/len(df)*100:.1f}%)")
    print(f"Dry races: {len(dry_races)} ({len(dry_races)/len(df)*100:.1f}%)")
    
    if len(wet_races) > 0:
        print("\nWet race winners (most frequent):")
        wet_winners = wet_races[wet_races['finish_pos'] == 1]
        if len(wet_winners) > 0:
            # Find driver columns that are 1 (winner)
            for _, row in wet_winners.iterrows():
                for col in df.columns:
                    if col.startswith('driver_') and row[col] == 1:
                        driver = col.replace('driver_', '')
                        print(f"  {driver}")
                        break

if __name__ == "__main__":
    print("🌧️  ENHANCED WEATHER FEATURE ENGINEERING")
    print("=" * 50)
    
    # Check weather data
    analyze_weather_impact()
    
    # Create enhanced features
    enhanced_df, features = enhanced_feature_engineering_with_weather()
    
    print(f"\n✅ Weather-enhanced features ready!")
    print(f"Key additions:")
    print(f"- Rain detection (wet/light/heavy)")
    print(f"- Temperature extremes")
    print(f"- Wet weather specialists")
    print(f"- Circuit-weather interactions")
    
    print(f"\nNext: python train_rank_model.py")