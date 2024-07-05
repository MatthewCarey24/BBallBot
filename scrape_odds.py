import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd

def scrape_odds(base_url, total_pages):
    all_data = []

    # Configure Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run Chrome in headless mode
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    for page_num in range(2, total_pages + 1):
        page_url = f"{base_url}{page_num}"
        print(f"Scraping page {page_num} - URL: {page_url}")

        # Open a new tab
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[page_num-2])

        # Fetch page content with Selenium
        driver.get(page_url)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(8)  # Adjust sleep time as needed for content to load
        
        # Parse page content with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find all match containers
        match_containers = soup.find_all(class_='eventRow flex w-full flex-col text-xs')
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

                # Extracting odds dynamically
                odds_elements = match_container.find_all('p', class_='height-content')
                odds_data = [odd.text.strip() for odd in odds_elements]

                if not odds_data:
                    print("No odds data found in this container")
                    continue
                if len(team_data) >= 2 and len(odds_data) >= 2:
                    home_team, away_team = team_data[0][0], team_data[1][0]
                    home_score, away_score = team_data[0][1], team_data[1][1]
                    home_odds, away_odds = odds_data[0], odds_data[1]
                    all_data.append([match_date, home_team, away_team, home_score, away_score, home_odds, away_odds])

                # Print or use the extracted data
                for idx, (team_name, team_score) in enumerate(team_data):
                    # Ensure there are enough odds to avoid IndexError
                    odds_text = odds_data[idx] if idx < len(odds_data) else 'N/A'
                    print(f"Team {idx + 1}: {team_name} ({team_score}) - Odds: {odds_text}")
                print("-----------------------------")

    driver.quit()  # Close the Selenium WebDriver
    return all_data

def main():
    base_url = 'https://www.oddsportal.com/basketball/usa/nba/results/#/page/'
    total_pages = 27  # Update with actual number of pages
    odds_data = scrape_odds(base_url, total_pages)

    if odds_data:
        df = pd.DataFrame(odds_data, columns=['Match Date','Home Team', 'Away Team', 'Home Score', 'Away Score', 'Home Odds', 'Away Odds'])
        df.to_csv('odds_data.csv', index=False)
        print("Data saved to odds_data.csv")
    else:
        print("No data scraped. Check the scraping process.")

if __name__ == '__main__':
    main()
