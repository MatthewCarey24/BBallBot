from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time

def scroll_and_wait(driver):
    """Scroll from top to bottom and wait for more content."""
    # Scroll to top first, just in case we're not at the top of the page
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)  # Allow some time for the page to adjust after scroll
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scroll a little bit down each time, 2/3 of the page height
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight / 3);")
        time.sleep(1)  # Wait for content to load
        
        # Calculate new scroll height and compare with last height
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # If the new height is the same as the last, we assume no more content is loading
        if new_height == last_height:
            time.sleep(1)  # Allow some extra time before breaking
            break
            
        last_height = new_height
        # Print progress
        print("Scrolling to load more matches...")


import pandas as pd
import os
import argparse

def process_page(driver, wait, page_number):
    """Process a single page of matches and return the match count."""
    match_count = 0
    page_data = []  # List to store match data for DataFrame
    current_date = None  # Track the current date
    
    try:
        # Scroll to load all matches
        print("\nScrolling to load all matches...")
        scroll_and_wait(driver)
        
        # Wait for all match containers to be present and visible (visibility check)
        print("Waiting for match containers...")
        time.sleep(1)
        # Wait for page to load and event rows to be visible
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'div[class*="eventRow"]')
            )
        )
        time.sleep(1)  # Give dynamic content time to load
        
        # Get all containers after waiting
        containers = driver.find_elements(By.CSS_SELECTOR, 'div[class*="eventRow"]')
        time.sleep(1)
        # Print number of containers found
        print(f"\nFound {len(containers)} match containers")
        
        # Parse page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # First find all event rows which might contain dates
        event_rows = soup.find_all('div', class_='eventRow flex w-full flex-col text-xs')
        
        print(f"\nFound {len(event_rows)} event rows")
        
        # Process each event row
        for row in event_rows:
            try:
                # Look for date in the event row
                # First check for date
                date_container = row.find('div', class_='border-black-borders bg-gray-light flex w-full min-w-0 border-l border-r')
                if date_container:
                    date_div = date_container.find('div', class_='text-black-main font-main w-full truncate text-xs font-normal leading-5')
                    if date_div:
                        current_date = date_div.text.strip()
                        print(f"Found date: {current_date}")
                
                # Find the game container within this row (could be in same container as date)
                container = row.find(class_='group flex')
                if not container:
                    continue

                # Find teams and scores
                away_team = container.find('a', class_='justify-content')
                home_team = container.find('a', class_='min-mt:!justify-end')
                
                if away_team and home_team:
                    # Get team names
                    away_name = away_team.find('p', class_='participant-name').text.strip()
                    home_name = home_team.find('p', class_='participant-name').text.strip()
                    
                    # Get scores
                    away_score = away_team.find('div', class_='min-mt:!hidden').text.strip()
                    home_score = home_team.find('div', class_='min-mt:!hidden').text.strip()
                    
                    # Find odds
                    odds_elements = container.find_all('p', class_='height-content')
                    odds = [elem.text.strip() for elem in odds_elements if elem.text.strip()]
                                        
                    # Find the div with class 'flex basis-[10%]'
                    div_container = container.find('div', class_="flex basis-[10%]")
                    # If found, get the <p> tag inside it and extract the text
                    game_time = div_container.find('p').text.strip() if div_container and div_container.find('p') else "N/A"                    
                    if odds and len(odds) >= 2:
                        home_odds = odds[0]  # Assuming the first odds value is for the home team
                        away_odds = odds[1]  # Assuming the second odds value is for the away team
                        
                        match_count += 1
                        # Store match data in page_data list with date
                        page_data.append([current_date, game_time, home_name, away_name, home_score, away_score, home_odds, away_odds])
                        print(f"\nMatch #{match_count} on page {page_number}: {home_name} ({home_score}) vs {away_name} ({away_score})")
                        print(f"Date: {current_date}")
                        print(f"Time: {game_time}")
                        print(f"Home Odds: {home_odds}, Away Odds: {away_odds}")
            except Exception as e:
                print(f"Error processing a match: {str(e)}")
                continue  # Continue to next container on error
                
    except Exception as e:
        print(f"Error processing page {page_number}: {str(e)}")
    
    return page_data, match_count

def save_to_csv(all_data, year):
    """Save all match data to a CSV file using pandas DataFrame."""
    # Create directory if it doesn't exist
    os.makedirs('odds_data', exist_ok=True)
    
    # Define CSV file path
    file_path = f'odds_data/odds_data_{year}.csv'
    
    # Create a DataFrame from all match data
    df = pd.DataFrame(all_data, columns=['Date', 'Time', 'Home Team', 'Away Team', 'Home Score', 'Away Score', 'Home Odds', 'Away Odds'])
    
    # Filter out playoff and pre-season games
    df = df[~df['Date'].str.contains('Play Offs|Pre-season', case=False, na=False)]
    
    # Reverse the entire dataset at once
    df = df.iloc[::-1]
    
    # Save the complete DataFrame
    df.to_csv(file_path, index=False)
    
    print(f"\nSaved {len(df)} matches to {file_path}")

def enhanced_scrape(year):
    # Basic Chrome options
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    
    # Initialize driver
    service = Service("C:/Users/mac22/OneDrive/Documents/BBallBot/node_modules/chromedriver-win64/chromedriver-win64/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 20)
    
    total_matches = 0
    all_data = []  # List to store all match data
    
    try:
        base_url = f'https://www.oddsportal.com/basketball/usa/nba-{year-1}-{year}/results/' if year != 2025 else 'https://www.oddsportal.com/basketball/usa/nba/results/'
        
        # Only process one page for testing
        for page in range(2, 28):
            url = f"{base_url}#/page/{page}/"
            print(f"\nAccessing page {page}: {url}")
            driver.get(url)
            driver.refresh()
            
            # Wait for page load
            time.sleep(1)  # Give the page time to load
            
            # Process the page and add to total matches
            page_data, matches_on_page = process_page(driver, wait, page)
            all_data.extend(page_data)  # Add this page's data to all_data
            if matches_on_page == 0:
                break
            total_matches += matches_on_page
            print(f"\nProcessed {matches_on_page} matches on page {page}")
            print(f"Total matches so far: {total_matches}")
        
        # Save all data at once after collecting everything
        save_to_csv(all_data, year)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nClosing browser...")
        driver.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape NBA odds data for a specific year')
    parser.add_argument('year', type=int, help='The year to scrape data for (e.g., 2021 for 2020-2021 season)')
    args = parser.parse_args()
    enhanced_scrape(args.year)
