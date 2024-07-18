import copy
import os
import numpy as np
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeries
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination import get_termination
from Optimization.problem import CounterfactualProblem
import matplotlib.pyplot as plt
from Counterfactuals.utils.counterfactual_utils import convert_string_to_array
from multiprocessing import Pool
import keras

class CONFETTI():
    def __init__(self, model_path, X_train, X_test, y_test, y_train, weights):
        self.model_path = model_path
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        model = keras.models.load_model(self.model_path)
        self.y_pred = np.argmax(model.predict(self.X_test), axis=1)
        self.y_pred_train = np.argmax(model.predict(self.X_train), axis=1)
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
        """
        Find the Nearest Unlike Neighbour (nun) for each instance in the Test Dataset

        Args:
            query: the instance for which we want to find its nun
            predicted_label = the classifier's predicted label for the query
            distance = distance metric to be used for the KNeighborsTimeSeries 
            n_neighbors = number of neighbors to be retrieved
        """
        df = pd.DataFrame(self.y_train, columns = ['label'])
        df.index.name = 'index'

        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)

        #Fit the KNN Algorithm with those instances that ARE NOT labelled as the Predicted Label    
        knn.fit(self.X_train[list(df[df['label'] != predicted_label].index.values)])
    
        dist, ind = knn.kneighbors(query.reshape(1,query.shape[0], query.shape[1]), return_distance=True)
        return dist[0], df[df['label'] != predicted_label].index[ind[0][:]]
    
    def naive_approach(self, instance, nun, model, subarray_length=1):
        """
        Swap the original instance's values for those of the nearest unlike neighbour (nun). Naive Approach. 

        Args:
            instance: the index of the original instance in the Test dataset
            nun: the index of the nearest unlike neighbour in the Train dataset
            subarray_length = The length of the sub-sequence that will be modified 
        """

        print('naive', instance, nun, model, subarray_length)
        #Initalize values
        most_influencial_array = self.findSubarray(self.weights[nun], subarray_length)

        starting_point = np.where(self.weights[nun]==most_influencial_array[0])[0][0]

        counterfactual = np.concatenate((self.X_test[instance][:starting_point], (self.X_train[nun][starting_point:subarray_length+starting_point]), 
                                        self.X_test[instance][subarray_length+starting_point:]))
        
        prob_target = model.predict(counterfactual.reshape(1,counterfactual.shape[0], counterfactual.shape[1]))[0][self.y_pred_train[nun]]

        
        
        while prob_target <= 0.5:
            subarray_length +=1
    
            #Get the sub-sequence.    
            most_influencial_array = self.findSubarray((self.weights[nun]), subarray_length)
            #Timestep where it starts
            starting_point = np.where(self.weights[nun]==most_influencial_array[0])[0][0]
            
            #Create the counterfactual by swapping the original instance's values for the NUN's.
            counterfactual = np.concatenate((self.X_test[instance][:starting_point], 
                                             (self.X_train[nun][starting_point:subarray_length+starting_point]), 
                                            self.X_test[instance][subarray_length+starting_point:]))
            
            #Feed new instance to model and check if the probability target changed. 
            prob_target = model.predict(counterfactual.reshape(1,counterfactual.shape[0], counterfactual.shape[1]))[0][self.y_pred_train[nun]]
        
        sparsity = np.mean(self.X_test[instance].flatten() == counterfactual.flatten())
        counterfactual_dict = {'Solution': [counterfactual], 'Window': subarray_length, 'Sparsity': sparsity, 'Precision': prob_target, 'Test Instance':instance, 'NUN Instance':nun}
        counterfactual_df = pd.DataFrame(counterfactual_dict)
        return counterfactual_df
    
    def optimization(self, instance_index:int, nun_index:int, subsequence_length:int, model):
        print('optimization', instance_index, nun_index, subsequence_length, model)
        #Initialize Values
        query = copy.deepcopy(self.X_test[instance_index])
        #starting_point = self.counterfactuals[instance_index][1]
        #end_point = starting_point + subsequence_length
        nun = copy.deepcopy(self.X_train[nun_index])
        solutions = pd.DataFrame(columns=["Solution", "Window", "Sparsity", "Precision", "Test Instance", "NUN Instance"])
        no_solution = 0

        #Start Optimization Search
        for window in range(subsequence_length, 1, -1): 
            print('window:', window)
            most_influencial_array = self.findSubarray((self.weights[nun_index]), window)
            #Timestep where it starts
            starting_point = np.where(self.weights[nun_index]==most_influencial_array[0])[0][0]
            end_point = starting_point + window

            #Define the Counterfactual Problem
            problem = CounterfactualProblem(query, nun, nun_index, starting_point, window, model, self.y_pred_train)
        
            # create the reference directions to be used for the optimization in NSGA3
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
            
            #NGSA-III Algorithm
            
            algorithm = NSGA3(pop_size=100,
                    ref_dirs = ref_dirs,
                    sampling=BinaryRandomSampling(),
                    crossover=TwoPointCrossover(),
                    mutation=BitflipMutation(),
                    )
            
        
            #Only do 300 generations
            termination = get_termination("n_gen", 300)
        
            #Run Optimization
            res = minimize(problem,algorithm,termination,seed=1,verbose=False)

            #Check if the optimization actually gave a solution
            if res.X is None:
                no_solution = no_solution + 1
            else:
                for x in res.X:
                    x_reshaped = res.X[0].reshape(window, query.shape[1])
                    op_counterfactual = np.copy(query)
                    op_counterfactual[starting_point:end_point][x_reshaped] = nun[starting_point:end_point][x_reshaped]  

                    op_counterfactual_reshaped = op_counterfactual.reshape(1,op_counterfactual.shape[0], op_counterfactual.shape[1])
                    f1 = model.predict(op_counterfactual_reshaped)[0][self.y_pred_train[nun_index]]
                    f2 = np.mean(query.flatten() == op_counterfactual.flatten())
                    row_dict = {'Solution': [op_counterfactual], 'Window': window, 'Sparsity': f2, 'Precision': f1, 'Test Instance':instance_index, 'NUN Instance': nun_index}
                    row_df = pd.DataFrame(row_dict)
                    solutions = pd.concat([solutions, row_df], ignore_index=True)
            
            if no_solution == 3:
                break
            
        return solutions
    
    
    def one_pass(self, test_instance):
        model = keras.models.load_model(self.model_path)
        nun = self.nearest_unlike_neighbour(self.X_test[test_instance], self.y_pred[test_instance], 'euclidean', 1)[1][0]
        naives = self.naive_approach(test_instance, nun, model)
        ce_optimized = self.optimization(test_instance,nun,naives.iloc[0]['Window'], model)
        
        return (nun, naives, ce_optimized) 
        
        
    def counterfactual_generator(self, directory = os.getcwd(), save_counterfactuals = False, optimization=False):
        self.naive_counterfactuals = pd.DataFrame(columns=["Solution", "Window", "Sparsity", "Precision", "Test Instance", "NUN Instance"])
        
        
        if optimization == True:
            #Optimize Counterfactuals
            self.optimized_counterfactuals = pd.DataFrame(columns=["Solution", "Window", "Sparsity", "Precision", "Test Instance", "NUN Instance"])
        
        pool = Pool(processes=8)
        res = pool.map(self.one_pass, range(len(self.X_test)))
        pool.close()
        pool.join()
        
        for r in res:
            print(r)
            #r = self.one_pass(i)
            self.nuns.append(r[0])
            self.naive_counterfactuals = pd.concat([self.naive_counterfactuals, r[1]], ignore_index=True)
            if optimization == True:
                self.optimized_counterfactuals = pd.concat([self.optimized_counterfactuals, r[2]], ignore_index=True)
                                    
        if optimization == True:
            self.optimized_counterfactuals = self.optimized_counterfactuals.groupby('Test Instance', as_index=False).apply(lambda x: x.drop_duplicates(subset='Window')).reset_index(drop=True)

        if save_counterfactuals == True:
            self.naive_counterfactuals.to_csv(str(directory+'_naive_counterfactuals.csv'), index=False)
            if optimization == True:
                self.optimized_counterfactuals.to_csv(str(directory+'_optimized_counterfactuals.csv'), index=False)
  
    def get_naive_counterfactual(self, instance:int):
        if hasattr(self, 'naive_counterfactuals'):
            return self.naive_counterfactuals[self.naive_counterfactuals["Test Instance"]==instance].iloc[0][0]
        else:
            raise AttributeError("Counterfactuals attribute does not exist yet. Please first run the function 'counterfactual_generator")          

    def get_optimized_counterfactual(self, instance:int, precision:bool):
        if hasattr(self, 'optimized_counterfactuals'):
            #Get the counterfactuals for instance x
            ce_instance = self.optimized_counterfactuals[self.optimized_counterfactuals["Test Instance"]==instance]
            if precision == True:
                #Return the counterfactual with highest precision
                return ce_instance.sort_values("Precision", ascending=False).iloc[0][0]
            else:
                #Return the counterfactual with highest sparsity
                return ce_instance.sort_values("Sparsity", ascending=False).iloc[0][0]
        else:
            raise AttributeError("Optimized Counterfactuals have not been generated yet. Run 'counterfactual_generator' with arg 'optimization' as True")

    def visualize_counterfactuals(self, instance:int, optimized=False, precision=True):
        sample = self.X_test[instance]
        if optimized == False:
            counterfactual = self.get_naive_counterfactual(instance)
        else:
            counterfactual = self.get_optimized_counterfactual(instance, precision)

            
        # Reshape Time Series for consistency with usual time series format [time, dimension]
        sample_reshaped = sample.T
        counterfactual_reshaped = counterfactual.T

        # Determine the number of dimensions (subplots) based on the input series
        num_dimensions = sample_reshaped.shape[0]

        # Set the style
        plt.style.use('seaborn-darkgrid') 
            
        # Create a plot with a dynamic number of subplots for the time series
        fig, axes = plt.subplots(num_dimensions, 1, figsize=(15, 3 * num_dimensions))

        # Iterate through each dimension
        for i, ax in enumerate(axes if num_dimensions > 1 else [axes]):
            # Plot the original series
            original_line, = ax.plot(sample_reshaped[i], label='Original', color='skyblue', linewidth=2)

            # Find the indices where the original and counterfactual differ
            diffs = np.where(sample_reshaped[i] != counterfactual_reshaped[i])[0]

            if diffs.size > 0:  # If there are differences
                # Determine the start and end of the continuous differing sequence
                start_diff = diffs[0]
                end_diff = diffs[-1]

                # Plot the entire sub-sequence where differences occur in salmon color
                counterfactual_line, = ax.plot(range(start_diff, end_diff + 1), 
                                            counterfactual_reshaped[i, start_diff:end_diff + 1], 
                                            label='Counterfactual', color='salmon', linewidth=2, linestyle='--')
                
                # Connecting line: from the last unchanged point to the first changed point
                start_diff = diffs[0]  # Start of the differing subsequence
                if start_diff > 0:  # Make sure there's a point to connect from
                    ax.plot([start_diff - 1, start_diff], [sample_reshaped[i, start_diff - 1], counterfactual_reshaped[i, start_diff]], color='salmon', linewidth=2)
                    
                # Connecting line: from the last changed point to the next original point
                end_diff = diffs[-1]  # End of the differing subsequence
                if end_diff < sample_reshaped[i].size - 1:  # Make sure there's a point to connect to
                    ax.plot([end_diff, end_diff + 1], [counterfactual_reshaped[i, end_diff], sample_reshaped[i, end_diff + 1]], color='salmon', linewidth=2)

            
            # Set legend manually to ensure no duplicate labels
            if diffs.size > 0:
                ax.legend(handles=[original_line, counterfactual_line], fontsize=12)
            else:
                ax.legend(handles=[original_line], fontsize=12)

            ax.set_title(f"Dimension {i+1}", fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True)  # Add gridlines

        plt.tight_layout()
        plt.show()

    def visualize_marks(self, instance:int, optimized=False, precision=True):
        sample = self.X_test[instance]
        if optimized == False:
            counterfactual = self.get_naive_counterfactual(instance)
        else:
            counterfactual = self.get_optimized_counterfactual(instance, precision)
            
        # Reshape Time Series for consistency with usual time series format [time, dimension]
        sample_reshaped = sample.T
        counterfactual_reshaped = counterfactual.T

        # Determine the number of dimensions (subplots) based on the input series
        num_dimensions = sample_reshaped.shape[0]

        # Set the style
        plt.style.use('seaborn-darkgrid')  # This applies a nice grid and background color
            
        # Create a plot with a dynamic number of subplots for the time series
        fig, axes = plt.subplots(num_dimensions, 1, figsize=(15, 3 * num_dimensions), sharex=True)

        # Iterate through each dimension
        for i, ax in enumerate(axes if num_dimensions > 1 else [axes]):
            # Plot where the original and counterfactual are the same
            same_indices = np.where(sample_reshaped[i] == counterfactual_reshaped[i])[0]
            diff_indices = np.where(sample_reshaped[i] != counterfactual_reshaped[i])[0]

            # Plot with dots
            if same_indices.size > 0:
                ax.plot(same_indices, sample_reshaped[i][same_indices], 'o', label='Same Values (OG)', color='blue', markersize=5)

            if diff_indices.size > 0:
                ax.plot(diff_indices, counterfactual_reshaped[i][diff_indices], 'o', label='Different Values (CE)', color='salmon', markersize=5)
                ax.plot(diff_indices, sample_reshaped[i][diff_indices], 'x', label='Different Values (OG)', color='green', markersize=5)
            # Set labels and legends
            ax.set_title(f"Dimension {i+1}", fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=12)
            ax.grid(True)  # Add gridlines

        plt.tight_layout()
        plt.show()

    def load_optimized_solutions(self, directory:str):
        self.optimized_counterfactuals = pd.read_csv(directory)
        self.optimized_counterfactuals["Solution"] = self.optimized_counterfactuals["Solution"].apply(
        lambda x: convert_string_to_array(x, timesteps=self.X_train.shape[1], channels=self.X_train.shape[2])
)
        