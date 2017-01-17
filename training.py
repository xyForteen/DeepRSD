# -*- coding: utf-8 -*-
"""
# Solar Energy Prediction with RNN
# 
# THIS file is for RNN training
"""
import gc
import os
#import sys
import time
import lasagne
import lasagne.layers as LL
from lasagne.objectives import aggregate
from lasagne.random import set_rng #, get_rng
#from lasagne.regularization import regularize_layer_params, l1
from sklearn.cross_validation import train_test_split
from math import log
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from RNN_arch import build_1Dregression_v1
#from trainingDataGen import generate_features

featureDim = 77
timeSteps = 5

rnd_SEED = 164141

# rewrite the mon, station-id feature
def rewrite(m, std):
    mon = [(i+1) for i in range(12)]
    mavg = np.mean(mon).astype(np.float32)
    mstd = np.std(mon).astype(np.float32)
    m[76, 0] = mavg
    std[76, 0] = mstd
   
    return m, std

# add noise to the training feature
# shape 501074*77*5
# add noise to 2nd-d, 
# 17-30, 32-45, 47-60, 62-75
def add_noise(X):
    for i in range(4):
        start = i*15 + 17
        end = i*15 +31
        X[:, start:end, :] += 0.01*np.random.randn(14, 5)
    return X

# preprocessing data, minus mean and devide variance
def transform(trainX, trainY):
    tX = np.reshape(trainX, (98*5113, featureDim, timeSteps))
    tY = np.reshape(trainY, (98*5113,), order = 'F')
    t1 = np.transpose(tX, (0, 2, 1))
    t2 = np.reshape(t1, (t1.shape[0]*t1.shape[1], featureDim))
    meanv = np.mean(t2, axis = 0, dtype = np.float64).astype(np.float32)
    m0 = np.atleast_2d(meanv).T
    stdv = np.std(t2, axis = 0, dtype = np.float64).astype(np.float32)
    std0 = np.atleast_2d(stdv).T
    m0, std0 = rewrite(m0, std0)
    tX -= m0
    tX /= std0
    print("Shape:", tX.shape)
    print("Data Type:", tX.dtype)
    np.save("mean.npy", m0)
    np.save("std.npy", std0)
    my = np.mean(tY, dtype = np.float64).astype(np.float32)
    stdy = np.std(tY, dtype = np.float64).astype(np.float32)
    # use original mean and std
    my = 16526160.0
    stdy = 7924210.0
    tY = (tY-my)/stdy
    print(my, stdy)
    return tX, tY
# split data for training and cross-validation
# trainX: 98*5113*Num*5
# trainY: 5113*98
def split_data(tX, tY):
    x_train = []
    x_cv = []
    y_train = []
    y_cv = []
    idx = np.random.permutation(5113*98)
    tX = tX[idx]
    tY = tY[idx]
    featureX = np.reshape(tX, (5113*98, featureDim*timeSteps))
    x_train, x_cv, y_train, y_cv = train_test_split(featureX, tY, 
                                                                test_size = .20, random_state = rnd_SEED)
    x_train = np.reshape(x_train, (x_train.shape[0], featureDim, timeSteps))
    #data augmentation
    #add some noise and permute samples
    '''
    x_train1 = add_noise(x_train0)
    x_train = np.vstack((x_train0, x_train1))
    y_train = np.hstack((y_train, y_train))
    num = x_train.shape[0]
    index = np.random.permutation(num)
    x_train = x_train[index]
    y_train = y_train[index]
    '''
    x_cv = np.reshape(x_cv, (x_cv.shape[0], featureDim, timeSteps))
       
    return x_train, x_cv, y_train, y_cv
    
    # tune momentum wrt. the number of mini-batches.    
def momentum_tune(num):
    umax = 0.995             # set umax to a conservative value
    n = num // 150 + 1
    p = log(n)/log(2)
    temp = 1 - 2 **(-1-p)
    return min(temp, umax)
    
def plot_training(t_scores, v_scores, starts, period):
    tt = np.array(t_scores) - 1
    tv = np.array(v_scores) - 1
    pt = np.trim_zeros(tt) + 1
    pv = np.trim_zeros(tv) + 1
    num = pt.shape[0]
    ends = starts + num*period
    plt.figure()
    plt.xlabel("Training batches")
    plt.ylabel("Loss")
    xx = np.linspace(starts, ends, num)
    ylim = (0.0, 1.0)
    plt.ylim(*ylim)
    plt.plot(xx, pt, '-', color="r", label = "training loss")
    plt.plot(xx, pv, '-', color="g", label = "validation loss")
    #plt.axis([20000, 32000, 0.2, 0.3])
    plt.legend(loc="best")
    plt.show()

############################### Main ################################
def do_regression(inputFeature = None,
                  truth = None,   # all the training and test data
                  destDirectory = None,
                  num_epochs=6, # No. of epochs to train
                  init_file=None,  # Saved parameters to initialise training
                  train_minibatch_size=64, 
                  valid_minibatch_size=100,
                  eval_multiple=100,
                  save_model=True,
                  input_width=timeSteps,
                  rng_seed=100009,
                  cross_val=0,  # Cross-validation subset label
                  dataver=1,  # Label for different runs/architectures/etc
                  rate_init=0.05,
                  rate_decay=0.999983,
                  #rate_decay = 1.0, 
                  momentum = 0.9):
    
    ###################################################
    ###### 1. Define model and theano variables                  #######
    ###################################################
    if rng_seed is not None:
        print("Setting RandomState with seed=%i" % (rng_seed))
        rng = np.random.RandomState(rng_seed)
        set_rng(rng)
    
    print("Defining variables...")
    index = T.lscalar() # Minibatch index
    x = T.tensor3('x') # Inputs 
    y = T.fvector('y') # Target
    
    print("Defining model...")
    network_0 = build_1Dregression_v1(
                        input_var=x,
                        input_width=input_width,
                        nin_units=64,
                        h_num_units=[128,224,128,16],
                        h_grad_clip=1.0,
                        output_width=1
                        )
                        
    if init_file is not None:
        print("Loading initial model parametrs...")
        init_model = np.load(init_file)
        init_params = init_model[init_model.files[0]]           
        LL.set_all_param_values([network_0], init_params)
    
    # Loading data generation model parameters
    print("Defining shared variables...")
    train_set_y = theano.shared(np.zeros(1, dtype=theano.config.floatX),
                                borrow=True) 
    train_set_x = theano.shared(np.zeros((1,1,1), dtype=theano.config.floatX),
                                borrow=True)
    
    valid_set_y = theano.shared(np.zeros(1, dtype=theano.config.floatX),
                                borrow=True)
    valid_set_x = theano.shared(np.zeros((1,1,1), dtype=theano.config.floatX),
                                borrow=True)
    '''                            
    x_train, x_cv, y_train, y_cv = split_data(trainX, trainY)
    train_set_y.set_value(y_train[:])
    train_set_x.set_value(x_train)
    valid_set_y.set_value(y_cv[:])
    valid_set_x.set_value(y_cv)
    '''
    ###################################################                                
    ########### 2. Create Loss expressions              ############
    ###################################################
    print("Defining loss expressions...")
    #r = lasagne.regularization.regularize_network_params(network_0, l1)
    prediction_0 = LL.get_output(network_0) 
    train_loss = aggregate(T.abs_(prediction_0 - y.dimshuffle(0,'x')))
    
    valid_prediction_0 = LL.get_output(network_0, deterministic=True)
    valid_loss = aggregate(T.abs_(valid_prediction_0 - y.dimshuffle(0,'x')))
    
    
    ###################################################                                
    ############ 3. Define update method  #############
    ###################################################
    print("Defining update choices...")
    params = LL.get_all_params(network_0, trainable=True)
    learn_rate = T.scalar('learn_rate', dtype=theano.config.floatX)
    momentum = T.scalar('momentum', dtype = theano.config.floatX)
    #updates = lasagne.updates.adadelta(train_loss, params,
                                      #learning_rate=learn_rate)
    updates = lasagne.updates.nesterov_momentum(train_loss, params,
                                      learning_rate=learn_rate, momentum = momentum)
    
    ###################################################                                
    ######### 4. Define train/valid functions                      #########
    ################################################### 
      
    print("Defining theano functions...")
    train_model = theano.function(
        [index, learn_rate, momentum],
        train_loss,
        updates=updates,
        givens={
            x: train_set_x[(index*train_minibatch_size):
                            ((index+1)*train_minibatch_size)],
            y: train_set_y[(index*train_minibatch_size):
                            ((index+1)*train_minibatch_size)]  
        }
    )
    
    validate_model = theano.function(
        [index],
        valid_loss,
        givens={
            x: valid_set_x[index*valid_minibatch_size:
                            (index+1)*valid_minibatch_size],
            y: valid_set_y[index*valid_minibatch_size:
                            (index+1)*valid_minibatch_size]
        }
    )
    
    ###################################################                                
    ################ 5. Begin training             ################
    ###################################################  
    print("Begin training...")
    
    this_train_loss = 0.0
    this_valid_loss = 0.0
    best_valid_loss = np.inf
    best_train_loss = np.inf
    
    eval_starts = 20000
    
    near_convergence = 30000
    eval_num = 10000
    train_eval_scores = np.ones(eval_num)
    valid_eval_scores = np.ones(eval_num)
    cum_iterations = 0
    eval_index = 0
    # store the best five models and scores
    models = list()
    scores = list()
    
    trainX = np.load(inputFeature)
    trainY = np.load(truth)
    trainX, trainY = transform(trainX, trainY)
    for i in range(num_epochs):
        x_train, x_cv, y_train, y_cv = split_data(trainX, trainY)
        train_batch_num = x_train.shape[0] // train_minibatch_size
        valid_batch_num = x_cv.shape[0] // valid_minibatch_size
        start_time = time.time()   
            
        train_set_y.set_value(y_train[:])
        train_set_x.set_value(x_train)
        valid_set_y.set_value(y_cv[:])
        valid_set_x.set_value(x_cv)
        
        # Iterate over minibatches in each batch
        for mini_index in xrange(train_batch_num):
            this_rate = np.float32(rate_init*(rate_decay**cum_iterations))
            # adaptive momentum
            this_momentum = 0.9
           
            if cum_iterations > near_convergence:
                this_rate = 0.001
               
            this_train_loss += train_model(mini_index, this_rate, this_momentum)
            cum_iterations += 1
            if np.isnan(this_train_loss):
                print "Training Error!!!!!!!!!"
                return
        # begin evaluation and report loss
            if (cum_iterations % eval_multiple == 0 and cum_iterations > eval_starts):
                this_train_loss = this_train_loss / eval_multiple
                this_valid_loss = np.mean([validate_model(k) for
                                    k in xrange(valid_batch_num)])
                train_eval_scores[eval_index] = this_train_loss
                valid_eval_scores[eval_index] = this_valid_loss
                
                # store models and scores
                if(len(scores) < 10):
                    scores.append(this_valid_loss)
                    models.append(LL.get_all_param_values(network_0))
                else:
                    scoreArr = np.array(scores)
                    idx = np.argmax(scoreArr)
                    if(this_valid_loss < scores[idx]):
                        scores[idx] = this_valid_loss
                        models[idx] = LL.get_all_param_values(network_0)
                # Save report every five evaluations
                if ((eval_index+1) % 10 == 0):
                    np.savetxt(
                       destDirectory + "train_score.txt",
                         train_eval_scores, fmt="%.5f"
                         )
                    np.savetxt(
                        destDirectory + "valid_score.txt",
                         valid_eval_scores, fmt="%.5f"
                         )
                # Save model if best validation score
                if (this_valid_loss < best_valid_loss):  
                    best_valid_loss = this_valid_loss
                    
                    if save_model:
                         filePath = destDirectory + "model"
                         np.savez(filePath, LL.get_all_param_values(network_0))
                # Reset evaluation reports
                print("Training Loss:", this_train_loss)
                print("Validation Loss:", this_valid_loss)
                print("Batches:", cum_iterations)
                print("Current learning rate:", this_rate)
                #print("Current momentum:", this_momentum)
                eval_index += 1
                this_train_loss = 0.0
                this_valid_loss = 0.0
            
        end_time = time.time()
        print("Computing time for epoch %d: %f" % (i, end_time-start_time))
        cur_train_loss = np.min(train_eval_scores)
        cur_valid_loss = np.min(valid_eval_scores)
        print("The best training loss in epoch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! %f" %cur_train_loss)
        print("The best validation loss in epoch!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! : %f" %cur_valid_loss)

    # save the best five scores
    path = destDirectory + "models\\"
    for n in range(10):
        filepath = path + "model" + str(n)
        np.savez(filepath, models[n])
    np.savetxt(path+"scores.txt", scores, fmt="%.5f")
    best_train_loss = np.min(train_eval_scores)
    best_valid_loss = np.min(valid_eval_scores)
    print("Best loss in training: %f" %best_train_loss)
    print("Best loss in cross-validation: %f" %best_valid_loss)
    plot_training(train_eval_scores, valid_eval_scores, eval_starts, eval_multiple)
 
    del train_set_x, train_set_y, valid_set_x, valid_set_y, trainX, trainY
    gc.collect()
    
    return None
    
def main(InputDirectory, DestDirectory):
        inputFeatFile = InputDirectory + "featureTrain.npy"
        inputTruth = InputDirectory + "truth.npy"
       
        do_regression(inputFeatFile, inputTruth, DestDirectory)

if __name__ == '__main__':
    os.chdir(r'E:\Projects\Solar energy prediction')
    args = {'InputDirectory' : 'E:\\Projects\\solar energy prediction\\',
    'DestDirectory' : 'E:\\Projects\\solar energy prediction\\'
    }
    main(**args)
         
            
                
            
            
            











