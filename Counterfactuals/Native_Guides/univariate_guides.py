import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Counterfactuals.utils.counterfactual_utils import ucr_data_loader
from tslearn.neighbors import KNeighborsTimeSeries
import keras

class CounterfactualGenerator:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.X_train, self.y_train, self.X_test, self.y_test = ucr_data_loader(str(dataset))
        self.nuns = []
        self.training_weights = np.load('/Users/alanparedescetina/Tesis/Counterfactuals/CAM/Weights/'+dataset+'_training_weights.npy')
        self.testing_weights = np.load('/Users/alanparedescetina/Tesis/Counterfactuals/CAM/Weights/'+dataset+'_testing_weights.npy')
        self.y_pred = np.argmax(self.model.predict(self.X_test), axis=1)

    def native_guide_retrieval(self, query, predicted_label, distance, n_neighbors):
        df = pd.DataFrame(self.y_train, columns = ['label'])
        df.index.name = 'index'
        
        df[df['label'] == 1].index.values, df[df['label'] != 1].index.values
        
        ts_length = self.X_train.shape[1]

        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)
        
        knn.fit(self.X_train[list(df[df['label'] != predicted_label].index.values)])
        
        dist,ind = knn.kneighbors(query.reshape(1,ts_length), return_distance=True)
        return dist[0], df[df['label'] != predicted_label].index[ind[0][:]]
    
    def findSubarray(a, k): #used to find the maximum contigious subarray of length k in the explanation weight vector
        
        n = len(a)
        
        vec=[] 

        # Iterate to find all the sub-arrays 
        for i in range(n-k+1): 
            temp=[] 

            # Store the sub-array elements in the array 
            for j in range(i,i+k): 
                temp.append(a[j]) 

            # Push the vector in the container 
            vec.append(temp) 

        sum_arr = []
        for v in vec:
            sum_arr.append(np.sum(v))

        return (vec[np.argmax(sum_arr)])
    
    def counterfactual_generator_swap(self, instance, nun, subarray_length):
        
        most_influencial_array= self.findSubarray((self.training_weights[nun]), subarray_length)
        
        starting_point = np.where(self.training_weights[nun]==most_influencial_array[0])[0][0]
        
        X_example = np.concatenate((self.X_test[instance][:starting_point], (self.X_train[nun][starting_point:subarray_length+starting_point]), 
                                    self.X_test[instance][subarray_length+starting_point:]))
        
        prob_target = self.model.predict(X_example.reshape(1,-1,1))[0][self.y_pred[instance]]
        
        while prob_target > 0.5:
            subarray_length +=1
            
            most_influencial_array = self.findSubarray((self.training_weights[nun]), subarray_length)
            starting_point = np.where(self.training_weights[nun]==most_influencial_array[0])[0][0]
            X_example = np.concatenate((self.X_test[instance][:starting_point], (self.X_train[nun][starting_point:subarray_length+starting_point]), 
                                        self.X_test[instance][subarray_length+starting_point:]))
            prob_target = self.model.predict(X_example.reshape(1,-1,1))[0][self.y_pred[instance]]
            
        return X_example
    
    def generate_counterfactuals(self):
        #Generate Next Unilikely Neighbour
        for instance in range(len(self.X_test)):
            self.nuns.append(self.native_guide_retrieval(self.X_test[instance], self.y_pred[instance], 'euclidean', 1)[1][0])
        self.nuns = np.array(self.nuns)

        test_instances = np.array(range(len(self.X_test)))
        self.cf_cam_swap = []
        for test_instance, nun in zip(test_instances, self.nuns):
            self.cf_cam_swap.append(self.counterfactual_generator_swap(test_instance, nun, 1))
        np.save(str('/Users/alanparedescetina/Tesis/Counterfactuals/Native_Guides/Guides/'+self.dataset) + '_native_guide_isw.npy', np.array(self.cf_cam_swap))

    def visualize_data(self, instance_index, labels):
        predicted_label = labels[np.argmax(self.model.predict(self.X_test[instance_index].reshape(1,-1,1)),axis=1)[0]]
        counterfactual_label = labels[np.argmax(self.model.predict(self.cf_cam_swap[instance_index].reshape(1,-1,1)),axis=1)[0]]

        df = pd.DataFrame({'Predicted: '+ predicted_label: list(self.X_test[instance_index].flatten()),
                           'Counterfactual: '+ counterfactual_label: list(self.cf_cam_swap[instance_index].flatten())})
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [
        '#08F7FE',  # teal/cyan
        '#FE53BB',  # pink
        '#F5D300',  # yellow
        '#00ff41',  # matrix green
        ]
        df.plot(marker='.', color=colors, ax=ax)
        n_shades = 10
        diff_linewidth = 1.05
        alpha_value = 0.3 / n_shades
        for n in range(1, n_shades+1):
            df.plot(marker='.',
                    linewidth=2+(diff_linewidth*n),
                    alpha=alpha_value,
                    legend=False,
                    ax=ax,
                    color=colors)
        ax.grid(color='#2A3459')
        plt.xlabel('Time', fontweight='bold', fontsize='large')
        plt.ylabel('Value', fontweight='bold', fontsize='large')
        plt.show()