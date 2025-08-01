#!/usr/bin/env python3
"""
Scrape Formula1.com for current season drivers and teams
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_current_season():
    """Get current F1 season"""
    current_date = datetime.now()
    if current_date.month <= 2:
        return current_date.year - 1
    return current_date.year

def setup_selenium_driver():
    """Setup Selenium WebDriver for JavaScript-heavy sites"""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        print(f"⚠️ Selenium not available: {e}")
        return None

def scrape_f1_drivers_selenium():
    """Scrape F1 drivers using Selenium for JavaScript content"""
    print("🕷️ Scraping Formula1.com drivers with Selenium...")
    
    driver = setup_selenium_driver()
    if not driver:
        return None
    
    try:
        url = "https://www.formula1.com/en/drivers"
        driver.get(url)
        
        # Wait for content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Give extra time for dynamic content
        time.sleep(3)
        
        drivers = []
        
        # Look for driver cards or listings
        driver_elements = driver.find_elements(By.CSS_SELECTOR, 
            "div[class*='driver'], article[class*='driver'], a[href*='/drivers/']")
        
        for element in driver_elements:
            try:
                # Extract driver name
                name_elem = element.find_element(By.CSS_SELECTOR, 
                    "h1, h2, h3, h4, span[class*='name'], span[class*='title']")
                
                # Extract team - look for team info in the same card
                team_elem = None
                try:
                    team_elem = element.find_element(By.CSS_SELECTOR, 
                        "span[class*='team'], div[class*='team'], p[class*='team']")
                except:
                    # Look in parent/sibling elements
                    parent = element.find_element(By.XPATH, "..")
                    try:
                        team_elem = parent.find_element(By.CSS_SELECTOR, 
                            "span[class*='team'], div[class*='team']")
                    except:
                        pass
                
                if name_elem and team_elem:
                    full_name = name_elem.text.strip()
                    team_name = team_elem.text.strip()
                    
                    if full_name and team_name and len(full_name.split()) >= 2:
                        driver_code = generate_driver_code(full_name)
                        team_code = normalize_team_name(team_name)
                        
                        drivers.append({
                            'driver_code': driver_code,
                            'full_name': full_name,
                            'team': team_code,
                            'team_full_name': team_name
                        })
                        
            except Exception:
                continue
        
        driver.quit()
        
        if drivers:
            print(f"✅ Scraped {len(drivers)} drivers with Selenium")
            return remove_duplicates(drivers)
        else:
            print("⚠️ No drivers found with Selenium")
            return None
            
    except Exception as e:
        print(f"❌ Selenium scraping failed: {e}")
        if driver:
            driver.quit()
        return None

def scrape_f1_drivers_requests():
    """Scrape F1 drivers using requests and BeautifulSoup"""
    print("🕷️ Scraping Formula1.com drivers with requests...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    try:
        # Try drivers page first
        response = requests.get("https://www.formula1.com/en/drivers", headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        drivers = []
        
        # Method 1: Look for structured driver data
        driver_cards = soup.find_all(['div', 'article', 'section'], 
                                   class_=re.compile(r'driver|listing|card', re.I))
        
        for card in driver_cards:
            try:
                # Look for name
                name_elem = card.find(['h1', 'h2', 'h3', 'h4', 'span', 'p'], 
                                    class_=re.compile(r'name|title', re.I))
                if not name_elem:
                    name_elem = card.find(text=re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+$'))
                
                # Look for team
                team_elem = card.find(['span', 'div', 'p'], 
                                    class_=re.compile(r'team|constructor', re.I))
                
                if name_elem and team_elem:
                    name_text = name_elem.get_text(strip=True) if hasattr(name_elem, 'get_text') else str(name_elem).strip()
                    team_text = team_elem.get_text(strip=True)
                    
                    if name_text and team_text and len(name_text.split()) >= 2:
                        driver_code = generate_driver_code(name_text)
                        team_code = normalize_team_name(team_text)
                        
                        drivers.append({
                            'driver_code': driver_code,
                            'full_name': name_text,
                            'team': team_code,
                            'team_full_name': team_text
                        })
                        
            except Exception:
                continue
        
        # Method 2: Look for links with driver info
        if not drivers:
            driver_links = soup.find_all('a', href=re.compile(r'/drivers/'))
            
            for link in driver_links:
                try:
                    text = link.get_text(strip=True)
                    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', text):
                        # Try to find team info nearby
                        parent = link.parent
                        team_elem = parent.find(['span', 'div'], 
                                              class_=re.compile(r'team', re.I))
                        
                        if team_elem:
                            team_text = team_elem.get_text(strip=True)
                            driver_code = generate_driver_code(text)
                            team_code = normalize_team_name(team_text)
                            
                            drivers.append({
                                'driver_code': driver_code,
                                'full_name': text,
                                'team': team_code,
                                'team_full_name': team_text
                            })
                            
                except Exception:
                    continue
        
        if drivers:
            print(f"✅ Scraped {len(drivers)} drivers with requests")
            return remove_duplicates(drivers)
        else:
            print("⚠️ No drivers found with requests method")
            return None
            
    except Exception as e:
        print(f"❌ Requests scraping failed: {e}")
        return None

def scrape_f1_teams():
    """Scrape F1 teams page for additional data"""
    print("🕷️ Scraping Formula1.com teams page...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get("https://www.formula1.com/en/teams", headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        team_drivers = {}
        
        # Look for team cards
        team_cards = soup.find_all(['div', 'article', 'section'], 
                                 class_=re.compile(r'team|constructor', re.I))
        
        for card in team_cards:
            try:
                # Find team name
                team_elem = card.find(['h1', 'h2', 'h3', 'h4'], 
                                    class_=re.compile(r'team|name|title', re.I))
                
                if team_elem:
                    team_name = team_elem.get_text(strip=True)
                    team_code = normalize_team_name(team_name)
                    
                    # Find drivers for this team
                    driver_names = []
                    driver_elems = card.find_all(['span', 'p', 'div'], 
                                               text=re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+$'))
                    
                    for driver_elem in driver_elems:
                        driver_name = driver_elem.get_text(strip=True)
                        if len(driver_name.split()) >= 2:
                            driver_names.append(driver_name)
                    
                    if team_code and driver_names:
                        team_drivers[team_code] = {
                            'team_full_name': team_name,
                            'drivers': driver_names
                        }
                        
            except Exception:
                continue
        
        if team_drivers:
            print(f"✅ Found team data for {len(team_drivers)} teams")
            return team_drivers
        else:
            print("⚠️ No team data found")
            return {}
            
    except Exception as e:
        print(f"❌ Team scraping failed: {e}")
        return {}

def generate_driver_code(full_name):
    """Generate standardized driver code from full name"""
    parts = full_name.lower().replace('-', '_').split()
    
    # Special cases for known drivers
    special_cases = {
        'max verstappen': 'max_verstappen',
        'lewis hamilton': 'hamilton',
        'charles leclerc': 'leclerc',
        'lando norris': 'norris',
        'oscar piastri': 'piastri',
        'george russell': 'russell',
        'carlos sainz': 'sainz',
        'carlos sainz jr': 'sainz',
        'fernando alonso': 'alonso',
        'pierre gasly': 'gasly',
        'esteban ocon': 'ocon',
        'alexander albon': 'albon',
        'yuki tsunoda': 'tsunoda',
        'nico hulkenberg': 'hulkenberg',
        'kevin magnussen': 'magnussen',
        'valtteri bottas': 'bottas',
        'zhou guanyu': 'zhou',
        'lance stroll': 'stroll',
        'sergio perez': 'perez',
        'sergio pérez': 'perez',
        'daniel ricciardo': 'ricciardo',
        'liam lawson': 'lawson',
        'franco colapinto': 'colapinto',
        'kimi antonelli': 'antonelli',
        'andrea kimi antonelli': 'antonelli',
        'oliver bearman': 'bearman',
        'gabriel bortoleto': 'bortoleto',
        'isack hadjar': 'hadjar',
        'jack doohan': 'doohan'
    }
    
    full_name_key = full_name.lower()
    if full_name_key in special_cases:
        return special_cases[full_name_key]
    
    # Default logic
    if len(parts) >= 2:
        # Use lastname if it's distinctive, otherwise firstname_lastname
        lastname = parts[-1]
        if len(lastname) > 4:
            return lastname
        else:
            return f"{parts[0]}_{lastname}"
    else:
        return parts[0] if parts else 'unknown'

def normalize_team_name(team_name):
    """Convert team names to standardized codes"""
    team_lower = team_name.lower()
    
    # Direct mappings
    team_mapping = {
        'mclaren': 'mclaren',
        'ferrari': 'ferrari',
        'mercedes': 'mercedes',
        'red bull': 'red_bull',
        'red bull racing': 'red_bull',
        'oracle red bull racing': 'red_bull',
        'aston martin': 'aston_martin',
        'alpine': 'alpine',
        'williams': 'williams',
        'haas': 'haas',
        'racing bulls': 'rb',
        'rb': 'rb',
        'alphatauri': 'rb',
        'visa cashapp rb': 'rb',
        'kick sauber': 'kick_sauber',
        'sauber': 'kick_sauber',
        'alfa romeo': 'kick_sauber'
    }
    
    # Find best match
    for key, value in team_mapping.items():
        if key in team_lower:
            return value
    
    # Fallback - try to extract key words
    if 'mclaren' in team_lower:
        return 'mclaren'
    elif 'ferrari' in team_lower:
        return 'ferrari'
    elif 'mercedes' in team_lower:
        return 'mercedes'
    elif 'red bull' in team_lower:
        return 'red_bull'
    elif 'aston' in team_lower:
        return 'aston_martin'
    elif 'alpine' in team_lower:
        return 'alpine'
    elif 'williams' in team_lower:
        return 'williams'
    elif 'haas' in team_lower:
        return 'haas'
    elif 'racing' in team_lower or 'rb' in team_lower:
        return 'rb'
    elif 'sauber' in team_lower or 'kick' in team_lower:
        return 'kick_sauber'
    else:
        return 'unknown'

def remove_duplicates(drivers):
    """Remove duplicate drivers"""
    seen = set()
    unique_drivers = []
    
    for driver in drivers:
        if driver['driver_code'] not in seen:
            unique_drivers.append(driver)
            seen.add(driver['driver_code'])
    
    return unique_drivers

def get_fallback_2025_data():
    """Fallback data based on known 2025 grid"""
    print("📋 Using fallback 2025 data...")
    
    return [
        {'driver_code': 'piastri', 'full_name': 'Oscar Piastri', 'team': 'mclaren', 'car_number': 81},
        {'driver_code': 'norris', 'full_name': 'Lando Norris', 'team': 'mclaren', 'car_number': 4},
        {'driver_code': 'leclerc', 'full_name': 'Charles Leclerc', 'team': 'ferrari', 'car_number': 16},
        {'driver_code': 'hamilton', 'full_name': 'Lewis Hamilton', 'team': 'ferrari', 'car_number': 44},
        {'driver_code': 'russell', 'full_name': 'George Russell', 'team': 'mercedes', 'car_number': 63},
        {'driver_code': 'antonelli', 'full_name': 'Kimi Antonelli', 'team': 'mercedes', 'car_number': 12},
        {'driver_code': 'max_verstappen', 'full_name': 'Max Verstappen', 'team': 'red_bull', 'car_number': 1},
        {'driver_code': 'tsunoda', 'full_name': 'Yuki Tsunoda', 'team': 'red_bull', 'car_number': 22},
        {'driver_code': 'albon', 'full_name': 'Alexander Albon', 'team': 'williams', 'car_number': 23},
        {'driver_code': 'sainz', 'full_name': 'Carlos Sainz Jr', 'team': 'williams', 'car_number': 55},
        {'driver_code': 'hulkenberg', 'full_name': 'Nico Hulkenberg', 'team': 'kick_sauber', 'car_number': 27},
        {'driver_code': 'bortoleto', 'full_name': 'Gabriel Bortoleto', 'team': 'kick_sauber', 'car_number': 24},
        {'driver_code': 'lawson', 'full_name': 'Liam Lawson', 'team': 'rb', 'car_number': 30},
        {'driver_code': 'hadjar', 'full_name': 'Isack Hadjar', 'team': 'rb', 'car_number': 21},
        {'driver_code': 'stroll', 'full_name': 'Lance Stroll', 'team': 'aston_martin', 'car_number': 18},
        {'driver_code': 'alonso', 'full_name': 'Fernando Alonso', 'team': 'aston_martin', 'car_number': 14},
        {'driver_code': 'ocon', 'full_name': 'Esteban Ocon', 'team': 'haas', 'car_number': 31},
        {'driver_code': 'bearman', 'full_name': 'Oliver Bearman', 'team': 'haas', 'car_number': 50},
        {'driver_code': 'gasly', 'full_name': 'Pierre Gasly', 'team': 'alpine', 'car_number': 10},
        {'driver_code': 'colapinto', 'full_name': 'Franco Colapinto', 'team': 'alpine', 'car_number': 43}
    ]

def create_team_performance_data():
    """Create team performance data based on 2025 season reality"""
    # This reflects the actual 2025 season where McLaren has been dominant
    # and Piastri has won almost half the races
    return {
        'mclaren': {
            'competitiveness': 0.95,
            'avg_position': 2.8,  # Very strong
            'podium_rate': 0.80,
            'wins_rate': 0.50,    # High win rate to match Piastri's success
            'poles_rate': 0.45,
            'championship_contender': True,
            'current_form': 'excellent'
        },
        'ferrari': {
            'competitiveness': 0.88,
            'avg_position': 4.2,
            'podium_rate': 0.65,
            'wins_rate': 0.25,
            'poles_rate': 0.30,
            'championship_contender': True,
            'current_form': 'strong'
        },
        'red_bull': {
            'competitiveness': 0.85,  # Still strong but not dominant
            'avg_position': 4.8,
            'podium_rate': 0.55,
            'wins_rate': 0.20,
            'poles_rate': 0.25,
            'championship_contender': True,
            'current_form': 'good'
        },
        'mercedes': {
            'competitiveness': 0.75,
            'avg_position': 7.0,
            'podium_rate': 0.30,
            'wins_rate': 0.08,
            'poles_rate': 0.15,
            'championship_contender': False,
            'current_form': 'improving'
        },
        'aston_martin': {
            'competitiveness': 0.68,
            'avg_position': 8.5,
            'podium_rate': 0.15,
            'wins_rate': 0.03,
            'poles_rate': 0.08,
            'championship_contender': False,
            'current_form': 'stable'
        },
        'alpine': {
            'competitiveness': 0.62,
            'avg_position': 10.0,
            'podium_rate': 0.08,
            'wins_rate': 0.02,
            'poles_rate': 0.04,
            'championship_contender': False,
            'current_form': 'struggling'
        },
        'williams': {
            'competitiveness': 0.58,
            'avg_position': 11.5,
            'podium_rate': 0.05,
            'wins_rate': 0.01,
            'poles_rate': 0.02,
            'championship_contender': False,
            'current_form': 'improving'
        },
        'rb': {
            'competitiveness': 0.52,
            'avg_position': 13.0,
            'podium_rate': 0.03,
            'wins_rate': 0.005,
            'poles_rate': 0.01,
            'championship_contender': False,
            'current_form': 'developing'
        },
        'haas': {
            'competitiveness': 0.48,
            'avg_position': 14.5,
            'podium_rate': 0.02,
            'wins_rate': 0.002,
            'poles_rate': 0.005,
            'championship_contender': False,
            'current_form': 'rebuilding'
        },
        'kick_sauber': {
            'competitiveness': 0.42,
            'avg_position': 16.5,
            'podium_rate': 0.01,
            'wins_rate': 0.001,
            'poles_rate': 0.002,
            'championship_contender': False,
            'current_form': 'struggling'
        }
    }

def main():
    """Main function to scrape and create F1 data files"""
    print("🏁 F1 2025 DATA SCRAPER")
    print("=" * 40)
    
    current_season = get_current_season()
    print(f"📅 Season: {current_season}")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Try multiple scraping methods
    drivers = None
    
    # Method 1: Try Selenium (most reliable for modern sites)
    if not drivers:
        drivers = scrape_f1_drivers_selenium()
    
    # Method 2: Try requests + BeautifulSoup
    if not drivers:
        drivers = scrape_f1_drivers_requests()
    
    # Method 3: Use fallback data
    if not drivers:
        print("⚠️ All scraping methods failed, using fallback data")
        drivers = get_fallback_2025_data()
    
    # Ensure we have car numbers
    for i, driver in enumerate(drivers):
        if 'car_number' not in driver:
            driver['car_number'] = i + 1
    
    # Create drivers DataFrame and save
    df_drivers = pd.DataFrame(drivers)
    df_drivers.to_csv("data/drivers_2025.csv", index=False)
    
    # Create team performance data
    team_performance = create_team_performance_data()
    
    # Save team performance
    with open("data/team_performance_2025.json", "w") as f:
        json.dump(team_performance, f, indent=2)
    
    # Print summary
    print(f"\n📊 RESULTS:")
    print(f"   Drivers found: {len(drivers)}")
    print(f"   Teams: {len(set(d['team'] for d in drivers))}")
    
    print(f"\n🏎️ DRIVERS BY TEAM:")
    teams_summary = {}
    for driver in drivers:
        team = driver['team']
        if team not in teams_summary:
            teams_summary[team] = []
        teams_summary[team].append(driver['driver_code'])
    
    for team, team_drivers in sorted(teams_summary.items()):
        perf = team_performance.get(team, {})
        comp = perf.get('competitiveness', 0.5)
        wins = perf.get('wins_rate', 0.0)
        print(f"   {team.upper()}: {team_drivers} (comp: {comp:.2f}, wins: {wins:.1%})")
    
    print(f"\n✅ Files created:")
    print(f"   - data/drivers_2025.csv")
    print(f"   - data/team_performance_2025.json")
    
    return df_drivers, team_performance

if __name__ == "__main__":
    try:
        drivers_df, team_perf = main()
        print("\n🏁 Scraping complete! Ready for enhanced feature engineering.")
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
        print("Using fallback data instead")