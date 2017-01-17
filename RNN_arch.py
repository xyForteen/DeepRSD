"""
# Solar Energy Prediction with RNN
# 
# THIS file is for the architeture of RNN
"""
import lasagne
import lasagne.layers as LL
import theano.tensor as T
import math

def build_1Dregression_v1(input_var=None, input_width=None, nin_units=48,
                            h_num_units=[64,64], h_grad_clip=1.0,
                            output_width=1):
    
    # Non-linearity hyperparameter
    leaky_ratio = 0.15
    nonlin = lasagne.nonlinearities.LeakyRectify(leakiness=leaky_ratio)
    
    # Input layer
    l_in = LL.InputLayer(shape=(None, 77, input_width), 
                            input_var=input_var) 
    batchsize = l_in.input_var.shape[0]
    
    # NIN-layer
    l_in = LL.NINLayer(l_in, num_units=nin_units,
                       nonlinearity=lasagne.nonlinearities.linear)
    #l_in_d = LL.DropoutLayer(l_in, p = 0.8) Do not use drop out now, for the first rnn layer is 256
    
    l_in_1 = LL.DimshuffleLayer(l_in, (0,2,1))
    
    # currently, we do not drop features
    
    # RNN layers
    # dropout in the first two (total three) or three (total five) layers
    initializer = lasagne.init.Normal()
    counter = -1
    drop_ends = 3
    for h in h_num_units:
        counter += 1
        # Forward layers
        l_forward_0 = LL.RecurrentLayer(l_in_1,
                                        nonlinearity=nonlin,
                                        num_units=h,
                                        W_in_to_hid=lasagne.init.Normal(0.01, 0),
                                        #W_in_to_hid=lasagne.init.He(initializer, math.sqrt(2/(1+0.15**2))),
                                        W_hid_to_hid=lasagne.init.Orthogonal(math.sqrt(2/(1+leaky_ratio**2))),
                                        backwards=False,
                                        learn_init=True,
                                        grad_clipping=h_grad_clip,
                                        #gradient_steps = 5,
                                        unroll_scan=True,
                                        precompute_input=True)
                                   
        l_forward_0a = LL.ReshapeLayer(l_forward_0, (-1, h))
        
        if(counter < drop_ends and counter % 2 != 0):
            l_forward_0a = LL.DropoutLayer(l_forward_0a, p = 0.5)
        else:
            l_forward_0a = l_forward_0a
        
        l_forward_0b = LL.DenseLayer(l_forward_0a, num_units=h,
                                     nonlinearity=nonlin)
        l_forward_0c = LL.ReshapeLayer(l_forward_0b,
                                       (batchsize, input_width, h))
#        
        if(counter < drop_ends):
            l_forward_out = LL.DropoutLayer(l_forward_0c, p = 0.3)
        else:
            l_forward_out = l_forward_0c
        
        '''# add noise 
        if(counter >1):
            l_forward_out = LL.GaussianNoiseLayer(l_forward_out, sigma=0.01)
        '''
        # Backward layers
        l_backward_0 = LL.RecurrentLayer(l_in_1,
                                         nonlinearity=nonlin,
                                         num_units=h,
                                         W_in_to_hid=lasagne.init.Normal(0.01, 0),
                                         #W_in_to_hid=lasagne.init.He(initializer, math.sqrt(2/(1+0.15**2))),
                                         W_hid_to_hid=lasagne.init.Orthogonal(math.sqrt(2/(1+leaky_ratio**2))),
                                         backwards=True,
                                         learn_init=True,
                                         grad_clipping=h_grad_clip,
                                         #gradient_steps = 5,
                                         unroll_scan=True,
                                         precompute_input=True)
                                        
        l_backward_0a = LL.ReshapeLayer(l_backward_0, (-1, h))
        
        if(counter < drop_ends and counter % 2 == 0):
            l_backward_0a = LL.DropoutLayer(l_backward_0a, p = 0.5)
        else:
            l_backward_0a = l_backward_0a
        
        l_backward_0b = LL.DenseLayer(l_backward_0a, num_units=h,
                                      nonlinearity=nonlin)
        l_backward_0c = LL.ReshapeLayer(l_backward_0b,
                                        (batchsize, input_width, h))
        
        if(counter < drop_ends ):
            l_backward_out = LL.DropoutLayer(l_backward_0c, p = 0.3)
        else:
            l_backward_out = l_backward_0c
        
        '''# add noise 
        if(counter >1):
            l_backward_out = LL.GaussianNoiseLayer(l_backward_out, sigma=0.01)
        '''
        l_in_1 = LL.ElemwiseSumLayer([l_forward_out, l_backward_out])
        #l_in_1 = LL.ElemwiseSumLayer([l_forward_0c, l_backward_0c])

        '''
        if counter < drop_ends:
            l_in_1 = LL.DropoutLayer(l_in_1_all, p = 0.3)
        else:
            l_in_1 = l_in_1_all
        '''            
                                                                                  
    # Output layers
    network_0a = LL.ReshapeLayer(l_in_1, (-1, h_num_units[-1]))
    network_0d = LL.DenseLayer(network_0a, num_units=output_width,
                               nonlinearity=nonlin)
    network_0e = LL.ReshapeLayer(network_0d,
                                 (batchsize, input_width, output_width))    
    
    output_net_1 = LL.FlattenLayer(network_0e, outdim=2)
    output_net_2 = LL.FeaturePoolLayer(output_net_1, pool_size=input_width,
                                       pool_function=T.mean)
    
    return output_net_2