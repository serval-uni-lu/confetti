import numpy as np
import keras
from keras.utils import to_categorical
from Models.classifiers import fcn
from Counterfactuals.CAM import class_activation_mapping as cam
from Counterfactuals.utils.counterfactual_utils import ucr_data_loader, visualize_series
from Counterfactuals.confetti import CONFETTI


from sktime.datasets import load_UCR_UEA_dataset
from Counterfactuals.utils.counterfactual_utils import label_encoder
from multiprocessing import Pool


def run_dataset(dataset):

	try:
		X_train, y_train = load_UCR_UEA_dataset(dataset, split="train", return_type="numpy3d")
		X_test, y_test = load_UCR_UEA_dataset(dataset, split="test", return_type="numpy3d")
	
	
		#Reshape data to (instances, timesteps, dimensions)
		X_train = X_train.reshape(X_train.shape[0],X_train.shape[2],X_train.shape[1])
		X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

		
		#Encode
		y_train, y_test = label_encoder(y_train, y_test)

		input_shape = X_train.shape[1:] #The input shape for our CNN should be (timesteps, dimensions)
		nb_classes = len(np.unique(np.concatenate([y_train,y_test])))

		#One Hot the labels for the CNN
		#y_train_encoded, y_test_encoded = to_categorical(y_train), to_categorical(y_test)

		# model = fcn.Classifier_FCN(
		# 	output_directory="./Models/",
		# 	input_shape=input_shape,
		# 	nb_classes= nb_classes,
		# 	dataset_name=dataset,
		# 	verbose=False
		# )

		# model.fit(X_train, y_train_encoded, X_test, y_test_encoded)

		model_path = './Models/'+dataset+'_best_model.hdf5'
		model = keras.models.load_model(model_path)

		weights_directory = './Counterfactuals/CAM/Weights/'
		testing_weights = cam.compute_weights_cam(model, X_test, dataset=dataset, save_weights=True, weights_directory=weights_directory, data_type='testing')

		ce = CONFETTI(model_path, X_train, X_test, y_test, y_train, testing_weights)


		import datetime
		print(dataset,'start', datetime.datetime.now())
		ce_directory = f'./Solutions/{dataset}'

		ce.counterfactual_generator(ce_directory, save_counterfactuals=True, optimization=True)
		print(dataset,'end', datetime.datetime.now())
	except:
		print('failed', dataset)


if __name__ == '__main__':

	datasets = [
		#'ArticularyWordRecognition',
		#'AtrialFibrillation',
		#'BasicMotions',
		
		#'PenDigits',
		#'RacketSports',
		#'LSST',
		#'Libras',
		#'FingerMovements',
		'NATOPS',
		'FaceDetection',
		'ERing',
		'PEMS-SF',
		'Epilepsy',
		'PhonemeSpectra',
		'DuckDuckGeese',
		'UWaveGestureLibrary',
		'HandMovementDirection',
		'Heartbeat',
		'SelfRegulationSCP1',
		'SelfRegulationSCP2',
		'Cricket',
		'EthanolConcentration',
		'StandWalkJump',
		'MotorImagery',
		'EigenWorms',


		#'Handwritting',
		#'InsectWingbeat',
		#'JapaneseVowels',
		#'SpokenArabicDigits',
		#'CharacterTrajectories',
	]

	for dataset in datasets:
		run_dataset(dataset)


		