"""Sleep Quality Predictor - Dataset 1 generation config. All data is SYNTHETIC."""
from datetime import datetime

N_PEOPLE = 300
N_DAYS = 30
RANDOM_SEED = 42
START_DATE = datetime(2026, 1, 5)  # a Monday

GENERATOR_VERSION = "1.0.0"
GENERATION_DATE = datetime.now().isoformat()
REFERENCE_DATASET = "Sleep_health_and_lifestyle_dataset.csv (Kaggle) - used for range calibration only, not copied"
