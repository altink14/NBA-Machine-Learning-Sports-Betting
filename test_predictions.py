import logging
from main_api import PredictionRunner

logging.basicConfig(level=logging.INFO)

try:
    print("Initializing PredictionRunner...")
    runner = PredictionRunner(sportsbook='fanduel', kelly_criterion=True)
    print("Running predictions...")
    res = runner.run_predictions()
    print("Result:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
