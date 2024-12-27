import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd

def scrape_odds(base_url, total_pages, max_retries=3):
    all_data = []

    # Configure Selenium WebDriver
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--no-sandbox")  # Recommended for headless environments
    options.add_argument("--disable-dev-shm-usage")  # Prevent issues in limited resource environments
    options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
    options.add_argument("--window-size=1920,1080")  # Set a larger window size
    options.add_argument("--start-maximized")  # Start with maximized window
    options.add_argument("--disable-notifications")  # Disable notifications
    options.add_argument("--disable-extensions")  # Disable extensions
    options.add_argument("--disable-infobars")  # Disable infobars
    options.add_argument("--blink-settings=imagesEnabled=false")  # Disable images for faster loading
    options.page_load_strategy = 'eager'  # Don't wait for all resources to load
    service = Service("C:/Users/carey/Desktop/BBallBot/node_modules/chromedriver/lib/chromedriver/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)

    for page_num in range(2, total_pages + 1):
        retries = 0
        while retries < max_retries:
            page_url = f"{base_url}{page_num}"
            print(f"Scraping page {page_num} - URL: {page_url}, Attempt: {retries + 1}")

            try:
                # Fetch page content with Selenium
                driver.get(page_url)
                driver.refresh()

                # Wait for initial page load with robust element checks
                wait = WebDriverWait(driver, 20)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'eventRow.flex.w-full.flex-col.text-xs')))
                wait.until(EC.visibility_of_element_located((By.CLASS_NAME, 'eventRow.flex.w-full.flex-col.text-xs')))
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'height-content')))  # Wait for odds elements
                
                # Additional wait for dynamic content
                driver.execute_script("""
                    return new Promise((resolve) => {
                        const checkReady = () => {
                            const elements = document.getElementsByClassName('eventRow flex w-full flex-col text-xs');
                            if (elements.length > 0 && Array.from(elements).every(el => el.offsetParent !== null)) {
                                resolve();
                            } else {
                                setTimeout(checkReady, 100);
                            }
                        };
                        checkReady();
                    });
                """)

                # Progressive scroll with dynamic wait
                last_height = driver.execute_script("return document.body.scrollHeight")
                match_containers = []
                scroll_attempts = 0
                max_scroll_attempts = 15

                while scroll_attempts < max_scroll_attempts:
                    # Scroll in smaller increments
                    current_height = driver.execute_script("return window.pageYOffset")
                    scroll_height = min(current_height + 800, last_height)  # Scroll 800px at a time
                    driver.execute_script(f"window.scrollTo(0, {scroll_height});")
                    
                    # Wait for dynamic content with exponential backoff
                    wait_time = min(0.5 * (1.5 ** scroll_attempts), 3)  # Cap at 3 seconds
                    time.sleep(wait_time)
                    
                    # Check for new elements with stale element handling
                    try:
                        match_containers = WebDriverWait(driver, 5).until(
                            EC.presence_of_all_elements_located((By.CLASS_NAME, 'eventRow flex w-full flex-col text-xs'))
                        )
                    except Exception:
                        # If wait times out, try direct find
                        match_containers = driver.find_elements(By.CLASS_NAME, 'eventRow flex w-full flex-col text-xs')
                    
                    # Check if we've reached the bottom
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height and len(match_containers) > 0:
                        # Double check with a final wait
                        time.sleep(1)
                        match_containers = driver.find_elements(By.CLASS_NAME, 'eventRow flex w-full flex-col text-xs')
                        break
                        
                    last_height = new_height
                    scroll_attempts += 1

                if len(match_containers) <= 20:  # Still using 20 as minimum threshold
                    print(f"Insufficient match containers found on page {page_num}. Retrying...")
                    retries += 1
                    continue  # Retry loading the page

                # Parse page content with BeautifulSoup
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Find all match containers
                match_containers = soup.find_all(class_='group flex')
                print(f"Match Containers: {len(match_containers)}")

                for idx, match_container in enumerate(match_containers):
                    print(f"Processing match container {idx + 1}")

                    date_element = match_container.find('div', class_='date time')
                    match_date = date_element.text.strip() if date_element else 'N/A'

                    # Extracting teams and scores dynamically
                    away_teams = match_container.find_all('a', class_='justify-content min-mt:!gap-2 flex basis-[50%] cursor-pointer items-center gap-1 overflow-hidden')
                    home_teams = match_container.find_all('a', class_='min-mt:!justify-end flex min-w-0 basis-[50%] cursor-pointer items-start justify-start gap-1 overflow-hidden')
                    teams = home_teams + away_teams
                    if not teams:
                        print("No teams found in this container")
                        continue

                    team_data = []
                    for team in teams:
                        team_name_element = team.find('p', class_='participant-name truncate')
                        team_score_element = team.find('div', class_='min-mt:!hidden')
                        team_name = team_name_element.text.strip() if team_name_element else 'N/A'
                        team_score = team_score_element.text.strip() if team_score_element else 'N/A'
                        team_data.append((team_name, team_score))

                    if not team_data:
                        print("No team data found in this container")
                        continue

                    # Extracting odds from the correct elements
                    odds_elements = match_container.find_all('p', class_='height-content')
                    odds_data = []
                    for odd in odds_elements:
                        print(odd)
                        odd_text = odd.text.strip()
                        if odd_text:
                            odds_data.append(odd_text)

                    if not odds_data:
                        print("No odds data found in this container")
                        continue

                    if len(team_data) >= 2 and len(odds_data) >= 2:
                        home_team, away_team = team_data[0][0], team_data[1][0]
                        home_score, away_score = team_data[0][1], team_data[1][1]
                        home_odds, away_odds = odds_data[0], odds_data[1]
                        all_data.append([match_date, home_team, away_team, home_score, away_score, home_odds, away_odds])

                    # Print extracted data
                    print(f"Date: {match_date}")
                    print(f"Home: {home_team} ({home_score}) - Odds: {home_odds}")
                    print(f"Away: {away_team} ({away_score}) - Odds: {away_odds}")
                    print("-----------------------------")

                break  # Exit the retry loop if scraping is successful

            except Exception as e:
                print(f"Error occurred on page {page_num}: {e}")
                retries += 1

    driver.quit()  # Close the Selenium WebDriver
    return all_data

def main():
    year = 2021
    base_url = f'https://www.oddsportal.com/basketball/usa/nba-{year-1}-{year}/results/#/page/'
    total_pages = 2  # Update with actual number of pages
    odds_data = scrape_odds(base_url, total_pages)

    if odds_data:
        df = pd.DataFrame(odds_data, columns=['Match Date', 'Home Team', 'Away Team', 'Home Score', 'Away Score', 'Home Odds', 'Away Odds'])
        df.to_csv(f'odds_data_{year}.csv', index=False)
        print(f"Data saved to odds_data_{year}.csv")
    else:
        print("No data scraped. Check the scraping process.")

if __name__ == '__main__':
    main()
