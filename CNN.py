from MultiLayerPerceptron import *
import math
import numpy as np
from scipy import ndimage
import time

class CNN(object):

    def __init__(self, layerSizes,weight_range:None, filter_size, stride, padding, weight_initialization="He"):
        
        self.MLP = MultiLayerPerceptron(layerSizes,None,"He")
        self.stride = stride
        self.padding = padding

        filter_count = 32
        weight_variance = 2/(filter_size*filter_size*filter_count)       
        self.conv_filters = np.random.randn(filter_count,filter_size,filter_size) * math.sqrt(weight_variance)
        self.conv_filters_biases = np.random.randn(filter_count,) * math.sqrt(weight_variance)
        
        """
        self.conv_filters = []
        for i in range(0,32):
            weight_variance = 2 / 1000          
            filterArray = math.sqrt(weight_variance) * np.random.randn(filter_size,filter_size)
            self.conv_filters.append(filterArray)
        """




    def feedforward(self, x, activation_function):

        first_x = x
        #CONVOLUTION
        height, width = x.shape
        filter_count, filter_height, filter_width = self.conv_filters.shape
           
        height_out = _compute_size(height, filter_height, self.padding, self.stride)
        width_out = _compute_size(width, filter_width, self.padding, self.stride)

        padding = (self.padding, self.padding)
        x_padded = np.pad(x, (padding, padding), mode='constant', constant_values=0)
        filters_reshaped = self.conv_filters.reshape(filter_count, -1).T

        out = np.zeros((filter_count, height_out, width_out))

        for h in range(height_out):
            for w in range(width_out):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + filter_height, w_start + filter_width
                window = x_padded[h_start: h_end, w_start: w_end]
                
                out[:, h, w] = np.dot(
                    window.reshape(1, -1),
                    filters_reshaped,
                ) + self.conv_filters_biases.T
        
        
        
        x = activation_function(out)
        conv_out = out

        #MAXPOOLING
        filter_count, height, width = x.shape
        height_out = _compute_size(height, 2, 0, 2)
        width_out = _compute_size(width, 2, 0, 2)
        

        

        out = np.zeros((filter_count, height_out, width_out))
        mask = np.zeros_like(x, dtype=np.uint8)
        for h in range(height_out):
            for w in range(width_out):
                h_start, w_start = h * 2, w * 2
                h_end, w_end = h_start + 2, w_start + 2
                window = x[:, h_start: h_end, w_start: w_end]
                
                flat_window = window.reshape(filter_count, -1)
               
                window_mask = flat_window.argmax(-1)[..., None] == range(flat_window.shape[-1])
                
                out[:, h, w] = flat_window[window_mask].reshape(filter_count)
                mask[:, h_start: h_end, w_start: w_end] = window_mask.reshape((filter_count, 2, 2))    

        
        
        
        return first_x, mask.astype(bool), conv_out, out



        


    def backpropagation(self,x,y,activation_function,derivative_function):
        
        first_x, mask, conv_out, mlpIN = self.feedforward(x,softplus_function)
        ynet,z_list,activations_list = self.MLP.feedforward(mlpIN.reshape(-1),softplus_function)
        fc_db, fc_dw = self.MLP.backpropagation(activations_list[0],y,softplus_function,sigmoid_function)
        
        delta =  self.MLP.weights[0].T@fc_db[0] * derivative_function(mlpIN.reshape(-1))
        delta = delta.reshape(mlpIN.shape)
        
        
        #upsample downstream
        d_downstream = np.zeros_like(conv_out)

        filter_count, height, width = delta.shape
        for h in range(height):
            for w in range(width):
                h_start, w_start = h * 2, w * 2
                h_end, w_end = h_start + 2, w_start + 2
                mask_window = mask[:, h_start: h_end, w_start: w_end]
                
                d_downstream[:,  h_start: h_end, w_start: w_end][mask_window] =  delta[:, h, w].flatten()

       
        #first layer error
        delta = d_downstream * conv_out
        

        filter_count, filter_height, filter_width = self.conv_filters.shape
        filters_reshaped = self.conv_filters.reshape(filter_count, -1)
        
        padding = (self.padding, self.padding)
        x_padded = np.pad(x, (padding, padding), mode='constant', constant_values=0)
        filters_reshaped = self.conv_filters.reshape(filter_count, -1).T
        d_downstream = np.zeros_like(x)
        d_weight = np.zeros_like(self.conv_filters)
        filter_count, height, width = delta.shape
       
        for h in range(height):
            for w in range(width):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + filter_height, w_start + filter_width           
                d_weight += (delta[:, h, w].T).reshape(filter_count,1).dot(
                    x_padded[h_start: h_end, w_start: w_end].reshape(1, -1)
                ).reshape(self.conv_filters.shape)

        d_bias = delta.sum(axis=(1, 2))
        

        return d_weight,d_bias, fc_dw, fc_db

    def train(self, training_data, validation_data, epochs, learn_step, minibatch_size, activation_function, derivative_function, patience):

        train_data_length = len(training_data[0])
        if validation_data:
            validation_data_length = len(validation_data[0])
        max_accuracy = 0.0
        max_epoch = epochs

        reversed_accuracy_list = []
        reversed_epoch_list = []

        for i in range(0,epochs):

           
            minibatches = [
                (training_data[0][j:j+minibatch_size],
                training_data[1][j:j+minibatch_size])
                for j in range(0, train_data_length, minibatch_size)]
            
            for minibatch in minibatches:
                
                gradient_f_weights = np.zeros_like(self.conv_filters)
                gradient_f_biases = np.zeros(self.conv_filters.shape[0])
                gradient_b = [np.zeros(b.shape) for b in self.MLP.biases]
                gradient_w = [np.zeros(w.shape) for w in self.MLP.weights]
                #print("EnterX:{0}".format(time.time()))
                minibatch_len = len(minibatch[0])
                for l in range(minibatch_len):
                    
                    x = minibatch[0][l].reshape(28,28) /255
                    y = minibatch[1][l]
                    
                    d_weight,d_bias, fc_dw, fc_db = self.backpropagation(x,y,activation_function,derivative_function)
                    
                    
                    gradient_f_weights += d_weight
                    gradient_f_biases += d_bias
                    gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, fc_db)]
                    gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, fc_dw)]

                #print("EntUpd:{0}".format(time.time()))
                self.MLP.weights = [w-(learn_step/minibatch_len)*nw
                        for w, nw in zip(self.MLP.weights, gradient_w)]
                self.MLP.biases = [b-(learn_step/minibatch_len)*nb
                       for b, nb in zip(self.MLP.biases, gradient_b)]
                self.conv_filters -= (learn_step/minibatch_len)* gradient_f_weights  
                self.conv_filters_biases -=  (learn_step/minibatch_len)* gradient_f_biases  
                
               # print(gradient_f_weights)
               # print(gradient_f_biases)
                #print(gradient_b)
                #print(gradient_w)
                     
                     
                    
            if True:
                #print(self.weights)
                #print(self.biases)
                #start = time.time()
                # print(self.conv_filters)
                #print("EntAcc:{0}".format(time.time()))
                accuracy = self.accuracy(validation_data,validation_data_length,activation_function)
               # print("FinEpo:{0}".format(time.time()))
                #print("VALIDATION TIME: {0}".format(time.time() - start))
                accuracy = round(accuracy,2)
                reversed_accuracy_list.insert(0,accuracy)
                reversed_epoch_list.insert(0,i)
                print ("Epoch {0}: {1}".format(
                    i, accuracy))
                np.set_printoptions(threshold=np.inf)
                
                  
            else:
                #print ("Epoch {0} complete".format(i))
                gg=False
                #return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
            if (accuracy > max_accuracy):
                #print("----------------{0}".format(i))
                max_epoch = i
                max_accuracy = accuracy
                
            if(i - max_epoch > patience):
                print("PATIENCE")
                
                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
                
       
        return epochs,reversed_epoch_list,reversed_accuracy_list,max_accuracy 

    def accuracy(self, validation_data, data_size, activation_function):

        hit_counter = 0

        for i in  range(data_size):
            x = validation_data[0][i].reshape(28,28) /255
            y = validation_data[1][i]
            _,_,_, mlpIN = self.feedforward(x,softplus_function)
            ynet,_,_ = self.MLP.feedforward(mlpIN.reshape(-1),softplus_function)
            
            if np.argmax(ynet) == y:
                hit_counter += 1
        return float(hit_counter) / data_size

    def feedforward_batch(self, x, activation_function):

        first_x = x
        #CONVOLUTION
        batch, height, width = x.shape
        filter_count, filter_height, filter_width = self.conv_filters.shape
           
        height_out = _compute_size(height, filter_height, self.padding, self.stride)
        width_out = _compute_size(width, filter_width, self.padding, self.stride)

        padding = (self.padding, self.padding)
        x_padded = np.pad(x, ((0,0), padding, padding), mode='constant', constant_values=0)
        
        
        filters_reshaped = self.conv_filters.reshape(filter_count, -1).T

        out = np.zeros((batch,filter_count, height_out, width_out))

        for h in range(height_out):
            for w in range(width_out):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + filter_height, w_start + filter_width
                window = x_padded[:, h_start: h_end, w_start: w_end]
                
                out[..., h, w] = np.dot(
                    window.reshape(batch, -1),
                    filters_reshaped,
                ) + self.conv_filters_biases.T
        np.set_printoptions(threshold=np.inf)
        
        x = activation_function(out)
        conv_out = out

        #MAXPOOLING
        batch,filter_count, height, width = x.shape
        height_out = _compute_size(height, 2, 0, 2)
        width_out = _compute_size(width, 2, 0, 2)
        

        

        out = np.zeros((batch, filter_count, height_out, width_out))
        mask = np.zeros_like(x, dtype=np.uint8)
        for h in range(height_out):
            for w in range(width_out):
                h_start, w_start = h * 2, w * 2
                h_end, w_end = h_start + 2, w_start + 2
                window = x[..., h_start: h_end, w_start: w_end]
                
                flat_window = window.reshape(*window.shape[:2], -1)
                window_mask = flat_window.argmax(-1)[..., None] == range(flat_window.shape[-1])
                out[..., h, w] = flat_window[window_mask].reshape(window.shape[:2])
                mask[..., h_start: h_end, w_start: w_end] = window_mask.reshape((*window.shape[:2], 2, 2))    

        
        
        
        return first_x, mask.astype(bool), conv_out, out


def _compute_size(image_size, filter_size, padding, stride):
    return 1 + (image_size + 2 * padding - filter_size) // stride





    