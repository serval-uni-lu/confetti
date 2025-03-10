from pathlib import Path

# Get the path of the current script file
current_file_path = Path(__file__).resolve()

BASE_DIR = current_file_path.parent
MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
CAM_WEIGHTS_DIR = TRAINED_MODELS_DIR
RESULTS_DIR = BASE_DIR / "results"

print(BASE_DIR)