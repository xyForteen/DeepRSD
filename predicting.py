"""
# Solar Energy Prediction with RNN
# Sep, 23th, 2016
# THIS file is for RNN prediction
"""

import csv
import os
import gc
import numpy as np
import lasagne.layers as LL
import theano
import theano.tensor as T

from RNN_arch import build_1Dregression_v1

# testX  98*1796*Num*5
# testX  98*1796*Num*5
def transform_data(testX):
    x_test =  np.reshape(testX, (98*1796, 77, 5))
    meanv = np.load('mean.npy').astype(np.float32)
    stdv = np.load('std.npy').astype(np.float32)
    x_test -= meanv
    x_test /= stdv
    return x_test
## or x_test =  np.reshape(testX, (98*1796, 77, 5), order = 'F')
    
############################### Main ################################
def do_prediction(testX,  # input features
                  destDirectory = None,
                  test_batch_num=None, # No. of minibatches per batch
                  test_minibatch_size=100,
                 # init_file=None,
                  input_width=5,
                  cross_val=0, # Cross-validation subset label
                  dataver=1): # Label for different runs/architectures/etc  
    
    ###################################################
    ############# 1. Housekeeping values ##############
    ###################################################
    test_batch_num = testX.shape[0]*testX.shape[1] // test_minibatch_size + 1
    
    ###################################################
    ###### 2. Define model and theano variables #######
    ###################################################
    print("Defining variables...")
    index = T.lscalar() # Minibatch index
    x = T.tensor3('x') # Inputs 
    
    print("Defining model...")
    network_0 = build_1Dregression_v1(
                        input_var=x,
                        input_width=input_width,
                        nin_units=64,
                        h_num_units=[128,224,128,16],
                        h_grad_clip=1.0,
                        output_width=1
                        )
    
    print("Setting model parametrs...")
    model_path = 'model.npz'  # the full path of model file 
    init_model = np.load(model_path)
    init_params = init_model[init_model.files[0]]           
    LL.set_all_param_values([network_0], init_params)
        
    ###################################################                                
    ################ 3. Import data ###################
    ###################################################
    ## Loading data generation model parameters
    print("Defining shared variables...")
    test_set_x = theano.shared(np.zeros((1,1,1), dtype=theano.config.floatX),
                               borrow=True)
    
    print("Assigning test data...")
    features = transform_data(testX)
    test_set_x.set_value(features)
    
    ###################################################                                
    ########### 4. Define prediction model ############
    ###################################################  
    print("Defining prediction expression...")
    test_prediction_0 = LL.get_output(network_0, deterministic=True)
    
    print("Defining theano functions...")
    test_model = theano.function(
        [index],
        test_prediction_0,
        givens={
            x: test_set_x[(index*test_minibatch_size):
                            ((index+1)*test_minibatch_size)],
        }
    )    
    
    ###################################################                                
    ############## 5. Begin predicting  ###############
    ###################################################  
    print("Begin predicting...")
    this_test_prediction= np.concatenate([test_model(i) for i in 
                                            xrange(test_batch_num)])

    ###################################################                                
    ################# 6. Save files  ##################
    ################################################### 
    print("Save submission file...") 
    predictions = np.reshape(this_test_prediction, (1796, 98), order = 'F')
    predictions = predictions * 7924210 + 16526158
    #predictions = predictions * 7924351 + 16526096
    fexample = open(r"Data\sampleSubmission.csv")  # file path of submission sample file
    fout = open(r"mySubmission.csv", 'wb')
    fReader = csv.reader(fexample,delimiter=',', skipinitialspace=True)
    fwriter = csv.writer(fout)
    for i,row in enumerate(fReader):
  		if i == 0:
 			fwriter.writerow(row)
  		else:
 			row[1:] = predictions[i-1]
 			fwriter.writerow(row)
    fexample.close()
    fout.close()

    del test_set_x
    gc.collect()
    
    return None
    

def main(InputDirectory, DestDirectory):
        
        os.chdir(r'E:\Projects\Solar energy prediction')
        inputFeatFile = InputDirectory + "featureTest.npy"
        trainX = np.load(inputFeatFile).astype(theano.config.floatX)
        do_prediction(trainX, DestDirectory)

if __name__ == '__main__':
    args = {'InputDirectory' : 'E:\\Projects\\solar energy prediction\\',
    'DestDirectory' : 'E:\\Projects\\solar energy prediction\\'
    }
    main(**args)