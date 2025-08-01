# debug_weather_startup.py - Debug why weather becomes empty during app startup
import os
import subprocess
import sys
import pandas as pd
from datetime import datetime

def check_weather_before_and_after():
    """Check weather file before and after running fetch_weather.py"""
    print("🔍 DEBUGGING WEATHER STARTUP ISSUE")
    print("=" * 60)
    
    weather_path = "data/race_weather.csv"
    
    def check_weather_file(stage):
        """Check weather file at different stages"""
        print(f"\n📊 WEATHER FILE STATUS - {stage}")
        print("-" * 40)
        
        if os.path.exists(weather_path):
            try:
                # Check file size
                file_size = os.path.getsize(weather_path)
                print(f"   File size: {file_size} bytes")
                
                if file_size == 0:
                    print(f"   ❌ File is completely empty")
                    return "empty"
                
                # Try to read
                df = pd.read_csv(weather_path)
                print(f"   Rows: {len(df)}")
                print(f"   Columns: {len(df.columns)}")
                
                if len(df.columns) > 0:
                    print(f"   Column names: {list(df.columns)}")
                    
                    if len(df) > 0:
                        print(f"   Sample data:")
                        for i, (_, row) in enumerate(df.head(3).iterrows()):
                            print(f"     Row {i+1}: {dict(row)}")
                        return "good"
                    else:
                        print(f"   ⚠️ Has columns but no data rows")
                        return "no_data"
                else:
                    print(f"   ❌ No columns")
                    return "no_columns"
                    
            except pd.errors.EmptyDataError:
                print(f"   ❌ Pandas says file is empty")
                return "pandas_empty"
            except Exception as e:
                print(f"   ❌ Error reading: {e}")
                return "error"
        else:
            print(f"   ❌ File doesn't exist")
            return "missing"
    
    # Check initial state
    initial_status = check_weather_file("BEFORE RUNNING FETCH_WEATHER")
    
    print(f"\n🔄 RUNNING fetch_weather.py...")
    print("-" * 40)
    
    try:
        # Run fetch_weather.py and capture output
        result = subprocess.run(
            [sys.executable, "fetch_weather.py"], 
            capture_output=True, 
            text=True, 
            timeout=180  # 3 minutes
        )
        
        print(f"   Return code: {result.returncode}")
        
        if result.stdout:
            print(f"   STDOUT (last 10 lines):")
            stdout_lines = result.stdout.strip().split('\n')
            for line in stdout_lines[-10:]:
                if line.strip():
                    print(f"     {line}")
        
        if result.stderr:
            print(f"   STDERR:")
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-5:]:
                if line.strip():
                    print(f"     {line}")
        
        # Check file after running
        after_status = check_weather_file("AFTER RUNNING FETCH_WEATHER")
        
        # Compare before and after
        print(f"\n📈 COMPARISON:")
        print(f"   Before: {initial_status}")
        print(f"   After:  {after_status}")
        
        if after_status == "good":
            print(f"   ✅ fetch_weather.py worked correctly")
        else:
            print(f"   ❌ fetch_weather.py didn't create proper data")
            
        return result.returncode == 0 and after_status == "good"
        
    except subprocess.TimeoutExpired:
        print(f"   ⏰ fetch_weather.py timed out after 3 minutes")
        return False
    except Exception as e:
        print(f"   ❌ Error running fetch_weather.py: {e}")
        return False

def simulate_app_startup():
    """Simulate what happens during app.py startup"""
    print(f"\n🚀 SIMULATING APP.PY STARTUP PROCESS")
    print("=" * 60)
    
    weather_path = "data/race_weather.csv"
    
    def run_step(step_name, script_name, timeout=120):
        """Run a startup step and check results"""
        print(f"\n{step_name}")
        print("-" * 40)
        
        # Check weather file before
        if os.path.exists(weather_path):
            before_size = os.path.getsize(weather_path)
            print(f"   Weather file before: {before_size} bytes")
        else:
            before_size = -1
            print(f"   Weather file before: doesn't exist")
        
        try:
            result = subprocess.run(
                [sys.executable, script_name], 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            print(f"   Return code: {result.returncode}")
            
            # Show last few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print(f"   Last output lines:")
                for line in lines[-3:]:
                    if line.strip():
                        print(f"     {line}")
            
            if result.stderr and result.returncode != 0:
                print(f"   Errors:")
                lines = result.stderr.strip().split('\n')
                for line in lines[-3:]:
                    if line.strip():
                        print(f"     {line}")
            
            # Check weather file after
            if os.path.exists(weather_path):
                after_size = os.path.getsize(weather_path)
                print(f"   Weather file after: {after_size} bytes")
                
                if before_size >= 0:
                    if after_size == 0 and before_size > 0:
                        print(f"   🚨 WARNING: Weather file became empty!")
                    elif after_size != before_size:
                        print(f"   📝 Weather file size changed: {before_size} → {after_size}")
                    else:
                        print(f"   ✅ Weather file size unchanged")
            else:
                print(f"   ❌ Weather file disappeared")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {script_name} timed out")
            return False
        except Exception as e:
            print(f"   ❌ Error running {script_name}: {e}")
            return False
    
    # Simulate the startup sequence from app.py
    steps = [
        ("1️⃣ FETCH RACES", "fetch_all_races.py"),
        ("2️⃣ FETCH WEATHER", "fetch_weather.py"),
        ("3️⃣ FEATURE ENGINEERING", "feature_engineering.py")
    ]
    
    for step_name, script in steps:
        success = run_step(step_name, script)
        
        if not success:
            print(f"\n❌ FAILED AT: {step_name}")
            break
        
        # Check if weather file is still good after each step
        if os.path.exists(weather_path):
            try:
                df = pd.read_csv(weather_path)
                if len(df) == 0 or len(df.columns) == 0:
                    print(f"\n🚨 WEATHER FILE BECAME EMPTY AFTER: {step_name}")
                    break
            except:
                print(f"\n🚨 WEATHER FILE BECAME CORRUPTED AFTER: {step_name}")
                break

def check_app_py_weather_handling():
    """Check how app.py handles weather subprocess"""
    print(f"\n🔍 CHECKING APP.PY WEATHER HANDLING")
    print("=" * 60)
    
    # Look at the relevant code in app.py
    try:
        with open("app.py", "r") as f:
            content = f.read()
        
        # Find the weather fetching section
        lines = content.split('\n')
        in_weather_section = False
        weather_code = []
        
        for i, line in enumerate(lines):
            if "Fetch latest weather data" in line or "fetch_weather.py" in line:
                in_weather_section = True
                # Get context around this line
                start = max(0, i - 3)
                end = min(len(lines), i + 10)
                weather_code.extend(lines[start:end])
                break
        
        if weather_code:
            print(f"📋 Weather fetching code in app.py:")
            for line in weather_code:
                print(f"   {line}")
        else:
            print(f"⚠️ Could not find weather fetching code in app.py")
            
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")

if __name__ == "__main__":
    print(f"🕐 Started at: {datetime.now()}")
    
    # First, test fetch_weather.py directly
    print(f"\n" + "=" * 80)
    print(f"STEP 1: TEST fetch_weather.py DIRECTLY")
    print(f"=" * 80)
    
    direct_success = check_weather_before_and_after()
    
    if direct_success:
        print(f"\n✅ fetch_weather.py works when run directly")
        
        # Now simulate the app startup process
        print(f"\n" + "=" * 80)
        print(f"STEP 2: SIMULATE APP STARTUP SEQUENCE")
        print(f"=" * 80)
        
        simulate_app_startup()
        
    else:
        print(f"\n❌ fetch_weather.py doesn't work when run directly")
        print(f"💡 Fix fetch_weather.py first before testing app startup")
    
    # Check app.py code
    check_app_py_weather_handling()
    
    print(f"\n🕐 Completed at: {datetime.now()}")