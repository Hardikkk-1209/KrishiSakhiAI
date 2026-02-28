import os
import pickle

def load_livestock_models(models_dir="trained_models"):

    models = {}

    required_files = {
        "health_predictor": "health_predictor.pkl",
        "anomaly_detector": "anomaly_detector.pkl",
        "gait_predictor": "gait_predictor.pkl",
        "disease_forecaster": "disease_forecaster.pkl"
    }

    for key, filename in required_files.items():
        path = os.path.join(models_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"{filename} not found in {models_dir}")

        with open(path, "rb") as f:
            models[key] = pickle.load(f)

    return models


# Dummy analyzers (if needed)
class GaitAnalyzer:
    def analyze_gait(self, data):
        return {"gait_score": data.get("gait_score", 0)}

class BehaviorAnalyzer:
    def analyze_behavior(self, data):
        return {"activity_level": data.get("activity_level", 0)}