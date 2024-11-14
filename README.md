# BBallBot

Machine learning model to predict the outcomes of NBA games

## Contains:
- ```scrape_odds.py```
scrapes the html of the website and formats game data in a csv for the model

- ```BBallBot.py```
processses the data, trains a model, and tests it on the end of the season. evaluates based on accuracy as well as potential profit using Kelly Criterion to inform betting quantities

## Future Work
- ```scrape_odds.py``` has trouble finding all the game html blocks on each site and can skip games, ensuring it consistently gets all data will allow it to be run without supervision
- the Kelly Criterion in ```BBallBot.py``` deems most bets not worth doing. Need to find optimal mathematical function for determining optimal percentage of wealth to donate
