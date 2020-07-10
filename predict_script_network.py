import numpy as np
import time
import os
import torch
import _pickle as cPickle
from src.RL import RL
from src.toric_model import Toric_code
from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################

# common system sizes are 3,5,7 and 9 
# grid size must be odd! 
system_size = 11

# valid network names: 
#   NN_11
#   NN_17
#   ResNet18
#   ResNet34
#   ResNet50
#   ResNet101
#   ResNet152
network = NN_11

# this file is stored in the network folder and contains the trained agent.  
NETWORK_FILE_NAME = 'size_11_size_11_NN_11_epoch_207'

num_of_predictions = 3000

# generate syndrome with a fixed amount of errors 
minimum_nbr_of_qubit_errors = int(system_size/2)+1 # minimum number of erorrs for logical qubit flip

# Generate folder structure, all results are stored in the data folder 
timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
PATH = 'data/NEWprediction__' +str(NETWORK_FILE_NAME) +'__'+  timestamp
if not os.path.exists(PATH):
    os.makedirs(PATH)

# Path for the network to use for the prediction
PATH2 = 'network/'+str(NETWORK_FILE_NAME)+'.pt'
print('Prediction')

# initial syndrome error generation 
# generate syndrome with error probability 0.1 
list_p_error = [0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19]
data_result = np.zeros((1, 2))
data_all = np.zeros((1, 8))

for p in list_p_error:

    prediction_list_p_error = [p]
    # initialize RL class
    rl = RL(Network=network,
            Network_name=NETWORK_FILE_NAME,
            system_size=system_size,
            device=device)

    error_corrected_list, ground_state_list, average_number_of_steps_list, failed_syndroms, prediction_list_p_error = rl.prediction(
        num_of_predictions=num_of_predictions, 
        num_of_steps=75, 
        PATH=PATH2,
        prediction_list_p_error=prediction_list_p_error,
        plot_one_episode=False)

    win_rate = (num_of_predictions - len(failed_syndroms)/2) / num_of_predictions

    # runtime of prediction
    runtime = time.time()-start
    runtime = runtime / 3600

    print(str(p), ' ', str(win_rate))
    #result 
    data_result = np.append(data_result, np.array([[p, win_rate]]), axis=0)
    np.savetxt(PATH + '/data_result.txt', data_result,  header='P_error, Win_rate', delimiter=',', fmt="%s")

    # save all
    data_all = np.append(data_all, np.array([[NETWORK_FILE_NAME, num_of_predictions, error_corrected_list, ground_state_list,average_number_of_steps_list, len(failed_syndroms)/2, win_rate, runtime]]), axis=0)
    np.savetxt(PATH + '/data_all.txt', data_all, header='network, error corrected, ground state conserved, average number of steps, number of failed syndroms, win_rate, runtime (h)', delimiter=',', fmt="%s")

# save training settings in txt file 
#data_all = np.array([[NETWORK_FILE_NAME, num_of_predictions, error_corrected_list, ground_state_list,average_number_of_steps_list, len(failed_syndroms)/2, win_rate, runtime]])
#np.savetxt(PATH + '/data_all.txt', data_all, header='network, error corrected, ground state conserved, average number of steps, number of failed syndroms, win_rate, runtime (h)', delimiter=',', fmt="%s")
