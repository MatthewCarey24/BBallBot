from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time

def enhanced_scrape():
    # Basic Chrome options
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    
    # Initialize driver
    service = Service("C:/Users/carey/Desktop/BBallBot/node_modules/chromedriver/lib/chromedriver/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        # Go to first page of results
        url = 'https://www.oddsportal.com/basketball/usa/nba-2023-2024/results/'
        print(f"Accessing URL: {url}")
        driver.get(url)
        
        # Initial wait for page load
        wait = WebDriverWait(driver, 20)
        
        # Function to scroll to bottom and wait for more content
        def scroll_and_wait():
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                # Scroll to bottom
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)  # Wait for content to load
                
                # Calculate new scroll height
                new_height = driver.execute_script("return document.body.scrollHeight")
                
                # Break if no more new content (height didn't change)
                if new_height == last_height:
                    break
                    
                last_height = new_height
                
                # Print progress
                print("Scrolling to load more matches...")
        
        # Scroll to load all matches
        print("\nScrolling to load all matches...")
        scroll_and_wait()
        
        # Wait for all match containers to be present
        print("Waiting for match containers...")
        containers = wait.until(
            EC.presence_of_all_elements_located(
                (By.CLASS_NAME, 'eventRow.flex.w-full.flex-col.text-xs')
            )
        )
        
        # Print number of containers found
        print(f"\nFound {len(containers)} match containers")
        
        # Parse page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        match_containers = soup.find_all(class_='group flex')
        
        print(f"\nFound {len(match_containers)} match containers in BeautifulSoup parsing")
        
        # Extract and print basic info from each container
        match_count = 0
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
                    
                    if odds:
                        match_count += 1
                        print(f"\nMatch #{match_count}: {home_name} ({home_score}) vs {away_name} ({away_score})")
                        print(f"Odds: {odds}")
            except Exception as e:
                print(f"Error processing a match: {str(e)}")
                continue  # Continue to next container on error
                
        print(f"\nTotal matches processed: {match_count}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nClosing browser...")
        driver.quit()

if __name__ == "__main__":
    enhanced_scrape()
