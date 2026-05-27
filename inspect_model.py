import xgboost as xgb
import json

def inspect_model(model_path):
    print(f"=== Inspecting {model_path} ===")
    booster = xgb.Booster()
    booster.load_model(model_path)
    # Check if booster has feature names
    print("Feature names in booster:", booster.feature_names)
    print("Feature types in booster:", booster.feature_types)

if __name__ == "__main__":
    inspect_model("Models/XGBoost_Models/XGBoost_68.9%_ML-3.json")
    inspect_model("Models/XGBoost_Models/XGBoost_54.8%_UO-8.json")
