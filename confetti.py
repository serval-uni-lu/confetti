import numpy as np
import keras
from sklearn import preprocessing
from pathlib import Path
import warnings
import confetti.CAM.class_activation_map as cam
from confetti.explainer.confetti_explainer import CONFETTI
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()
import time
if __name__ == "__main__":

    # Suppress specific warning
    warnings.filterwarnings("ignore", category=FutureWarning)

    from sktime.datasets import load_UCR_UEA_dataset
    dataset = 'BasicMotions'

    #Data will load with shape (instances, dimensions, timesteps)
    X_train, y_train = load_UCR_UEA_dataset("BasicMotions", split="train", return_type="numpy3d")
    X_test, y_test = load_UCR_UEA_dataset("BasicMotions", split="test", return_type="numpy3d")

    #Reshape data to (instances, timesteps, dimensions)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[2],X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    #Encode
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    #One Hot the labels for the CNN
    y_train_encoded, y_test_encoded = keras.utils.to_categorical(y_train), keras.utils.to_categorical(y_test)


    nb_classes = len(np.unique(np.concatenate([y_train,y_test])))
    input_shape = X_train.shape[1:] #The input shape for our CNN should be (timesteps, dimensions)


    #Load trained model
    # Ensure the path is always relative to the project root
    model_path = "/Users/alan.paredes/Desktop/confetti/models/trained_models/BasicMotions/BasicMotions_fcn.keras"
    # Load the model
    model = keras.models.load_model(str(model_path))



    testing_weights = cam.compute_weights_cam(model, X_test, dataset=dataset, save_weights=True, data_type='testing')



    ce = CONFETTI(model_path, X_train, X_test[:1], y_test[:1], y_train, testing_weights)

    ce_directory = Path.cwd().parent/ "results" / dataset
    ce.parallelized_counterfactual_generator(ce_directory,save_counterfactuals=True,processes=1)
