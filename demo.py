from pathlib import Path
import keras

from confetti import CONFETTI
from confetti.explainer import CounterfactualResults
from confetti.utils import load_multivariate_ts_from_csv
from confetti.attribution import cam
from confetti.visualizations import plot_time_series, plot_counterfactual

keras.utils.disable_interactive_logging()

ROOT = Path("examples")

DATA_SCRIPT = ROOT / "scripts" / "generate_toy_dataset.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_toy_model.py"

DATA_TRAIN_PATH = ROOT / "data" / "toy_train.csv"
DATA_TEST_PATH = ROOT / "data" / "toy_test.csv"
MODEL_PATH = ROOT / "models" / "toy_fcn.keras"

print("Scripts located:")
print(DATA_SCRIPT)
print(TRAIN_SCRIPT)

def demo():
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded:", MODEL_PATH)
    X_train, y_train = load_multivariate_ts_from_csv(str(DATA_TRAIN_PATH))
    X_test, y_test = load_multivariate_ts_from_csv(str(DATA_TEST_PATH))

    print("Dataset loaded.")
    print("Shape:", X_train.shape, y_train.shape)
    print("Example sample shape:", X_train[0].shape)


    training_weights = cam(model, X_train)

    instance_to_explain = X_test[0:1]
    plot_time_series(instance_to_explain[0], title="Instance to Explain")

    explainer = CONFETTI(model_path=MODEL_PATH)

    ce_results : CounterfactualResults = explainer.generate_counterfactuals(
        instances_to_explain=instance_to_explain,
        reference_data=X_train,
        reference_weights=training_weights,
        verbose=True
    )
    print("Counterfactual generation complete.")
    for sets in ce_results.counterfactual_sets:
        print("Original instance shape:", sets.original_instance.shape)
        print("Original Label:", sets.original_label)
        print("Best Counterfactual", sets.best.counterfactual.shape)
        print("Best Counterfactual Label:", sets.best.label)
        print("Nearest Unlike Neighbour shape:", sets.nearest_unlike_neighbour.shape)
        print("Training Weights used:", sets.feature_importance)

        plot_time_series(series=sets.nearest_unlike_neighbour, title="Nearest Unlike Neighbour")

        plot_counterfactual(
                original = sets.original_instance,
                counterfactual = sets.best,
                title = "Normal Counterfactual Plot",)

        plot_counterfactual(
                original = sets.original_instance,
                counterfactual = sets.best,
                channels = [0],
                title = "Selecting One Channel",)


        plot_counterfactual(
            original=sets.original_instance,
            counterfactual=sets.best,
            cam_weights=sets.feature_importance,
            cam_mode="line",
            title="Counterfactual with CAM as line", )

        plot_counterfactual(
            original=sets.original_instance,
            counterfactual=sets.best,
            cam_weights=sets.feature_importance,
            cam_mode="heatmap",
            title="Counterfactual with CAM as heatmap", )

def main():
    demo()

if __name__ == "__main__":
    main()

