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
    time.sleep(2)  # Allow some time for the page to adjust after scroll
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scroll a little bit down each time, 2/3 of the page height
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight / 3);")
        time.sleep(3)  # Wait for content to load
        
        # Calculate new scroll height and compare with last height
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # If the new height is the same as the last, we assume no more content is loading
        if new_height == last_height:
            time.sleep(2)  # Allow some extra time before breaking
            break
            
        last_height = new_height
        # Print progress
        print("Scrolling to load more matches...")


import pandas as pd
import os

def process_page(driver, wait, page_number, year, is_first_page=False):
    """Process a single page of matches and return the match count."""
    match_count = 0
    page_data = []  # List to store match data for DataFrame
    
    try:
        # Scroll to load all matches
        print("\nScrolling to load all matches...")
        scroll_and_wait(driver)
        
        # Wait for all match containers to be present and visible (visibility check)
        print("Waiting for match containers...")
        time.sleep(3)
        containers = wait.until(
            EC.visibility_of_all_elements_located(
                (By.CLASS_NAME, 'eventRow.flex.w-full.flex-col.text-xs')
            )
        )
        time.sleep(2)
        # Print number of containers found
        print(f"\nFound {len(containers)} match containers")
        
        # Parse page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        match_containers = soup.find_all(class_='group flex')
        
        print(f"\nFound {len(match_containers)} match containers in BeautifulSoup parsing")
        
        # Extract and print basic info from each container
        for container in match_containers:
            try:
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
                    
                    if odds and len(odds) >= 2:
                        home_odds = odds[0]  # Assuming the first odds value is for the home team
                        away_odds = odds[1]  # Assuming the second odds value is for the away team
                        
                        match_count += 1
                        # Store match data in page_data list
                        page_data.append([home_name, away_name, home_score, away_score, home_odds, away_odds])
                        print(f"\nMatch #{match_count} on page {page_number}: {home_name} ({home_score}) vs {away_name} ({away_score})")
                        print(f"Home Odds: {home_odds}, Away Odds: {away_odds}")
            except Exception as e:
                print(f"Error processing a match: {str(e)}")
                continue  # Continue to next container on error
                
    except Exception as e:
        print(f"Error processing page {page_number}: {str(e)}")
    
    # Save data to CSV after processing the page
    save_to_csv(page_data, year, is_first_page)
    
    return match_count

def save_to_csv(page_data, year, is_first_page=False):
    """Save match data to a CSV file using pandas DataFrame."""
    # Create directory if it doesn't exist
    os.makedirs('odds_data', exist_ok=True)
    
    # Define CSV file path
    file_path = f'odds_data/odds_data_{year}.csv'
    
    # Create a DataFrame from the match data
    df = pd.DataFrame(page_data, columns=['Home Team', 'Away Team', 'Home Score', 'Away Score', 'Home Odds', 'Away Odds'])
    
    # If it's the first page, overwrite the file. Otherwise, append.
    mode = 'w' if is_first_page else 'a'
    header = True if is_first_page else False
    df.to_csv(file_path, mode=mode, header=header, index=False)
    
    print(f"\nSaved {len(page_data)} matches to {file_path}")

def enhanced_scrape():
    # Basic Chrome options
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    
    # Initialize driver
    service = Service("C:/Users/carey/Desktop/BBallBot/node_modules/chromedriver/lib/chromedriver/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 20)
    
    total_matches = 0
    year = 2020  # Set year manually as you requested
    last_page = 27
    
    try:
        base_url = f'https://www.oddsportal.com/basketball/usa/nba-{year-1}-{year}/results/'
        
        # Process pages 1-24 in ascending order
        for page in range(2, last_page + 1):
            url = f"{base_url}#/page/{page}/"
            print(f"\nAccessing page {page}: {url}")
            driver.get(url)

            driver.refresh()
            
            # Wait for page load
            time.sleep(2)  # Give the page time to load
            
            # Process the page and add to total matches
            is_first_page = (page == 1)  # First page when starting from 1
            matches_on_page = process_page(driver, wait, page, year, is_first_page)
            total_matches += matches_on_page
            print(f"\nProcessed {matches_on_page} matches on page {page}")
            print(f"Total matches so far: {total_matches}")
        
        print(f"\nGrand total of matches processed across all pages: {total_matches}")
        
        # After scraping is complete, reverse the CSV order
        print("\nReversing CSV order...")
        csv_path = f'odds_data/odds_data_{year}.csv'
        df = pd.read_csv(csv_path)
        df_reversed = df.iloc[::-1]  # Reverse the DataFrame
        df_reversed.to_csv(csv_path, index=False)
        print("CSV order reversed successfully")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nClosing browser...")
        driver.quit()

if __name__ == "__main__":
    enhanced_scrape()
