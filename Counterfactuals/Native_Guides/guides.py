import numpy as np
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeries
import os
import matplotlib.pyplot as plt

class CounterfactualGenerator():
    def __init__(self, model, X_train, X_test, y_test, y_train, weights):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        self.weights = weights
        self.nuns = []

    def findSubarray(self, a: list, k: int): #used to find the maximum contigious subarray of length k in the explanation weight vector
        
        n = len(a)
            
        vec = [] 

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
    
    def nearest_unlike_neighbour(self, query, predicted_label, distance, n_neighbors):
        df = pd.DataFrame(self.y_train, columns = ['label'])
        df.index.name = 'index'

        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)

        #Fit the KNN Algorithm with those instances that ARE NOT labelled as the Predicted Label    
        knn.fit(self.X_train[list(df[df['label'] != predicted_label].index.values)])
    
        dist, ind = knn.kneighbors(query.reshape(1,query.shape[0], query.shape[1]), return_distance=True)
        return dist[0], df[df['label'] != predicted_label].index[ind[0][:]]
    
    def swap(self, instance, nun, subarray_length=1):

        most_influencial_array = self.findSubarray(self.weights[nun], subarray_length)

        starting_point = np.where(self.weights[nun]==most_influencial_array[0])[0][0]

        counterfactual = np.concatenate((self.X_test[instance][:starting_point], (self.X_train[nun][starting_point:subarray_length+starting_point]), 
                                        self.X_test[instance][subarray_length+starting_point:]))
        
        prob_target = self.model.predict(counterfactual.reshape(1,counterfactual.shape[0], counterfactual.shape[1]))[0][self.y_pred[instance]]
    
        while prob_target > 0.5:
            subarray_length +=1
                
            most_influencial_array = self.findSubarray((self.weights[nun]), subarray_length)
            starting_point = np.where(self.weights[nun]==most_influencial_array[0])[0][0]
            
            counterfactual = np.concatenate((self.X_test[instance][:starting_point], 
                                             (self.X_train[nun][starting_point:subarray_length+starting_point]), 
                                            self.X_test[instance][subarray_length+starting_point:]))
            
            prob_target = self.model.predict(counterfactual.reshape(1,counterfactual.shape[0],counterfactual.shape[1]))[0][self.y_pred[instance]]
      
        return counterfactual
    
    def counterfactual_generator(self, directory = os.getcwd(), save_counterfactuals = False):
        #Generate Next Unilikely Neighbour
        for instance in range(len(self.X_test)):
            self.nuns.append(self.nearest_unlike_neighbour(self.X_test[instance], self.y_pred[instance], 'euclidean', 1)[1][0])
        self.nuns = np.array(self.nuns)

        test_instances = np.array(range(len(self.X_test)))
        self.counterfactuals = []
        for test_instance, nun in zip(test_instances, self.nuns):
            self.counterfactuals.append(self.swap(test_instance, nun))
        
        if save_counterfactuals == True:
            np.save(directory, np.array(self.counterfactuals))
        
       
    def get_counterfactual(self, instance):
        return self.counterfactuals[instance]
                
        
    def visualize_counterfactuals(self, instance:int):
        sample = self.X_test[instance]
        counterfactual = self.get_counterfactual(instance)
        
        # Reshape Time Series for consistency with usual time series format [time, dimension]
        sample_reshaped = sample.T
        counterfactual_reshaped = counterfactual.T

        # Determine the number of dimensions (subplots) based on the input series
        num_dimensions = sample_reshaped.shape[0]

        # Set the style
        plt.style.use('seaborn-darkgrid')  # This applies a nice grid and background color
        
        # Create a plot with a dynamic number of subplots for the time series
        fig, axes = plt.subplots(num_dimensions, 1, figsize=(15, 3 * num_dimensions))

        # Iterate through each dimension
        for i, ax in enumerate(axes if num_dimensions > 1 else [axes]):
            # Plot the original series
            original_line, = ax.plot(sample_reshaped[i], label='Original', color='skyblue', linewidth=2)

            # Find the indices where the original and counterfactual differ
            diffs = np.where(sample_reshaped[i] != counterfactual_reshaped[i])[0]

            if diffs.size > 0:  # If there are differences
                # Plot the differing subsequence from the original time series
                changed_subsequence, = ax.plot(diffs, sample_reshaped[i, diffs], label='Changed Subsequence (Original)', color='lightgreen', linewidth=2, alpha=0.5)
                # Plot the differing subsequence from the counterfactual
                counterfactual_line, = ax.plot(diffs, counterfactual_reshaped[i, diffs], label='Counterfactual', color='salmon', linewidth=2, linestyle='--')

                # Connecting line: from the last unchanged point to the first changed point
                start_diff = diffs[0]  # Start of the differing subsequence
                if start_diff > 0:  # Make sure there's a point to connect from
                    ax.plot([start_diff - 1, start_diff], [sample_reshaped[i, start_diff - 1], counterfactual_reshaped[i, start_diff]], color='salmon', linewidth=2)
                
                # Connecting line: from the last changed point to the next original point
                end_diff = diffs[-1]  # End of the differing subsequence
                if end_diff < sample_reshaped[i].size - 1:  # Make sure there's a point to connect to
                    ax.plot([end_diff, end_diff + 1], [counterfactual_reshaped[i, end_diff], sample_reshaped[i, end_diff + 1]], color='salmon', linewidth=2)

            
            # Set legend manually to exclude the 'Changed Subsequence (Original)'
            ax.legend(handles=[original_line, counterfactual_line], fontsize=12)
            ax.set_title(f"Dimension {i+1}", fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True)  # Add gridlines

        plt.tight_layout()
        plt.show()
       