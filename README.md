# NBA Sports Betting Using Machine Learning 🏀
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

> **About this fork (Betting Buddy backend).** This repo began as a fork of [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting); much of the text below is the original project's. Two of its claims no longer describe this code and are corrected here:
>
> - **Accuracy.** The original README quoted a moneyline accuracy that was the best of many random train/test splits, not a measure of how the model does on games it has not seen, and this project does not use it. The model served here (`Models/candidate_2026-08`, XGBoost, calibrated) was evaluated once, sealed, on games played entirely after training: **67.2% moneyline accuracy across 2,597 games in 2024-25 and 2025-26** (95% CI 65.3-68.9%). Picking the home team every time would have gone 55.1%. Accuracy is not profitability; no return on investment has been measured.
> - **Over/under.** The totals pick was withdrawn on 2026-09-19: it never had a sealed evaluation. The market's total is still shown; our pick is not.
>
> Every prediction is logged before tip-off and graded in public. The API is FastAPI (`main_api.py`); see `DEPLOY.md`.

A machine learning model used to predict the winners of NBA games from team data matched with the odds of those games. Outputs expected value for teams money lines to provide better insight. The fraction of your bankroll to bet based on the Kelly Criterion is also outputted. Note that a popular, less risky approach is to bet 50% of the stake recommended by the Kelly Criterion.
## Packages Used

Use Python 3.11. In particular the packages/libraries used are...

* Tensorflow - Machine learning library
* XGBoost - Gradient boosting framework
* Numpy - Package for scientific computing in Python
* Pandas - Data manipulation and analysis
* Colorama - Color text output
* Tqdm - Progress bars
* Requests - Http library
* Scikit_learn - Machine learning library

## Usage

<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Expected_value.png" width="1010" height="424" />

Make sure all packages above are installed.

```bash
$ git clone https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting.git
$ cd NBA-Machine-Learning-Sports-Betting
$ pip3 install -r requirements.txt
$ python3 main.py -xgb -odds=fanduel
```

Odds data will be automatically fetched from sbrodds if the -odds option is provided with a sportsbook.  Options include: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny

If `-odds` is not given, enter the under/over and odds for today's games manually after starting the script.

Optionally, you can add '-kc' as a command line argument to see the recommended fraction of your bankroll to wager based on the model's edge

## Flask Web App

> *From the original project. There is no Flask app in this repo; the API is FastAPI: `venv/Scripts/python.exe -m uvicorn main_api:app --port 8000`.*

<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

This repo also includes a small Flask application to help view the data from this tool in the browser.  To run it:
```
cd Flask
flask --debug run
```

## Getting new data and training models
```
# Create dataset with the latest data for 2023-24 season
cd src/Process-Data
python -m Get_Data
python -m Get_Odds_Data
python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML
python -m XGBoost_Model_UO
```

## Contributing

All contributions welcomed and encouraged.
