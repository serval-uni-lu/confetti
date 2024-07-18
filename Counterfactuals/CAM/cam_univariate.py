import tensorflow as tf
from tensorflow import keras
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.interpolate import interp1d
from ..utils.counterfactual_utils import ucr_data_loader


def cam_plotter(dataset):
    
    """The dataset here will be the one which we traind the FCN on and saved the corresponding best model"""
    
    model_path = '/Users/alanparedescetina/Tesis/Models/'
    
    X_train, y_train, X_test, y_test = ucr_data_loader(dataset)
    
    
    model = keras.models.load_model(model_path+dataset+'_best_model.hdf5')

    max_length = 2000
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()

    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[2],X_train.shape[1]))

    w_k_c = model.layers[-1].get_weights()[0]  # weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    for c in classes: #the class
        plt.figure()
        count = 0
        #Get all the TS from X_train where the label correspond to class C
        c_x_train = X_train[np.where(y_train == c)]
        for ts in c_x_train:
            ts = ts.reshape(1, -1, 1)
            [conv_out, predicted] = new_feed_forward([ts])
            pred_label = np.argmax(predicted)
            orig_label = np.argmax(enc.transform([[c]]))
            if pred_label == orig_label:
                cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
                for k, w in enumerate(w_k_c[:, orig_label]):
                    cas += w * conv_out[0, :, k]

                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100

                x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                # linear interpolation to smooth
                f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                y = f(x)
                # if (y < -2.2).any():
                #     continue
                f = interp1d(range(ts.shape[1]), cas)
                cas = f(x).astype(int)
                plt.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=2, vmin=0, vmax=100, linewidths=0.0)
                #plt.ylabel('Signal',  fontsize='xx-large', fontweight='bold')

        cbar = plt.colorbar()
    #plt.savefig('../Images/' + str(dataset) +'.pdf')
        # cbar.ax.set_yticklabels([100,75,50,25,0])

### Training Weights
def training_weights_cam(dataset, save_weights):
    
    X_train, y_train, X_test, y_test = ucr_data_loader(dataset)

    model_path = '/Users/alanparedescetina/Tesis/Models/'
    weights_directory= '/Users/alanparedescetina/Tesis/Counterfactuals/CAM/Weights/'
    
    model = keras.models.load_model(model_path+dataset+'_best_model.hdf5')
    
    w_k_c = model.layers[-1].get_weights()[0]

    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    weights = []
    for i, ts in enumerate(X_train):
        ts = ts.reshape(1,-1,1)
        [conv_out, predicted] = new_feed_forward([ts])
        pred_label = np.argmax(predicted)

        cas =   np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
        for k, w in enumerate(w_k_c[:, pred_label]):
            cas += w * conv_out[0,:, k]
        weights.append(cas)
    weights = np.array(weights)
    
    if save_weights == True:

        np.save(weights_directory+dataset+'_training_weights.npy',weights)
        
    return weights

def testing_weights_cam(dataset, save_weights):
    
    X_train, y_train, X_test, y_test = ucr_data_loader(dataset)

    model_path = '/Users/alanparedescetina/Tesis/Models/'
    weights_directory= '/Users/alanparedescetina/Tesis/Counterfactuals/CAM/Weights/'
    
    model = keras.models.load_model(model_path+dataset+'_best_model.hdf5')
    
    
    w_k_c = model.layers[-1].get_weights()[0]

    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)


    weights = []
    for i, ts in enumerate(X_test):
        ts = ts.reshape(1,-1,1)
        [conv_out, predicted] = new_feed_forward([ts])
        pred_label = np.argmax(predicted)

        cas =   np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
        for k, w in enumerate(w_k_c[:, pred_label]):
            cas += w * conv_out[0,:, k]
        weights.append(cas)
    weights = np.array(weights)
    
    
    if save_weights == True:
        
        np.save(weights_directory+dataset+'_testing_weights.npy',weights)
        
    return weights