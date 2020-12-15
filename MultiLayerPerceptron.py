import random


import numpy as np
import time
import math

class MultiLayerPerceptron(object):

    def __init__(self, layerSizes,weight_range:None, weight_initialization):
        
        self.layerSizes = layerSizes
        self.weights = []
        self.biases = []
        for i in range(0,len(layerSizes)-1):

            if weight_range == None:
                #Xavier
                weight_variance = 0.1
                if weight_initialization == 'Xavier':
                    weight_variance = 2 / (layerSizes[i]+ layerSizes[i+1])

                #He
                if weight_initialization == 'He':
                    
                    weight_variance = 2 / layerSizes[i]            
                
               
                self.weights.append(math.sqrt(weight_variance) * np.random.randn(layerSizes[i+1],layerSizes[i]))
                self.biases.append(math.sqrt(weight_variance) * np.random.randn(layerSizes[i+1]))
            else:
                low,high = weight_range
                self.weights.append(np.random.uniform(low,high,(layerSizes[i+1],layerSizes[i])))
                self.biases.append(np.random.uniform(low,high,layerSizes[i+1]))
        


        
    def feedforward(self, input,  activation_function):

        z_list = []
        activations_list = [input]

        wb_size = len(self.weights)
        for i in range (0, wb_size-1):
            z = np.dot(self.weights[i], input)+self.biases[i]
            z_list.append(z)
            input = activation_function(z)
            activations_list.append(input)
            
        #calculate outer layer with softmax func
        z = np.dot(self.weights[wb_size-1], input)+self.biases[wb_size-1]
        z_list.append(z)
        input = softmax(z)
        activations_list.append(input)
        return input,z_list,activations_list

    def backpropagation(self, x, y, activation_function, derivative_function):

        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]

        # forward propagation, save activations and z

        net_y,z_list,activations_list = self.feedforward(x,activation_function)
        
        # outer layer error - activation function - softmax
        y_hotone = np.zeros(10)
        y_hotone[y] = 1
        delta = net_y - y_hotone
        
        
        gradient_biases[-1] = delta
        gradient_weights[-1] = np.outer(delta ,(activations_list[-2]).T)
    
        # error propagation
        
        """
        for i in range (len(self.weights)-2,-1):
            print("siema")
            z = z_list[i]
            act_deriv = derivative_function(z)
            print(self.weights[i+1].T)
            print(delta)
            print(act_deriv)
            delta = self.weights[i+1].T @ delta * act_deriv
            gradient_biases[i] = delta
            gradient_weights[i] = np.outer(delta, (activations_list[i]).T)
        """
        
        z = z_list[0]
        act_deriv = derivative_function(z)
        
        delta = self.weights[1].T @ delta * act_deriv
        gradient_biases[0] = delta
        
        gradient_weights[0] = np.outer(delta, (activations_list[0]).T)
        return gradient_biases,gradient_weights

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
             
                gradient_b = [np.zeros(b.shape) for b in self.biases]
                gradient_w = [np.zeros(w.shape) for w in self.weights]
               # start = time.time()
                minibatch_len = len(minibatch[0])
                for l in range(minibatch_len):
                    
                    x = minibatch[0][l]
                    y = minibatch[1][l]
                    #startT = time.time_ns()
                    delta_gradient_b, delta_gradient_w = self.backpropagation(x,y,activation_function,derivative_function)
                   # end = time.time_ns() - startT
                    gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
                    gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

                #print("BATCH TIME: {0}".format(time.time() - start))
                #print("{2},{3},PROP TO BATCH : {0} | {1}".format(end*minibatch_len/10**9,(time.time()-start),i,l))
               # start = time.time()
                self.weights = [w-(learn_step/minibatch_len)*nw
                        for w, nw in zip(self.weights, gradient_w)]
                self.biases = [b-(learn_step/minibatch_len)*nb
                       for b, nb in zip(self.biases, gradient_b)]
                #print("UPDATE TIME: {0}".format(time.time() - start))         
            if validation_data:
                #print(self.weights)
                #print(self.biases)
                #start = time.time()
                accuracy = self.accuracy(validation_data,validation_data_length,activation_function)
                #print("VALIDATION TIME: {0}".format(time.time() - start))
                accuracy = round(accuracy,2)
                reversed_accuracy_list.insert(0,accuracy)
                reversed_epoch_list.insert(0,i)
                print ("Epoch {0}: {1}".format(
                    i, accuracy))
            else:
                print ("Epoch {0} complete".format(i))

                return i,reversed_epoch_list,reversed_accuracy_list,max_accuracy
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
            x = validation_data[0][i]
            y = validation_data[1][i]
            net_y,_,_ = self.feedforward(x,activation_function)
            
            if np.argmax(net_y) == y:
                hit_counter += 1
        return float(hit_counter) / data_size

    def feedforward_batch(self, input,  activation_function):

        z_list = []
        activations_list = [input]

        wb_size = len(self.weights)
        for i in range (0, wb_size-1):
            z = np.dot(self.weights[i], input.T)+self.biases[i]
            z_list.append(z)
            input = activation_function(z)
            activations_list.append(input)
            
        #calculate outer layer with softmax func
        z = np.dot(self.weights[wb_size-1], input)+self.biases[wb_size-1]
        z_list.append(z)
        input = softmax(z)
        activations_list.append(input)
        return input,z_list,activations_list



    def save(self):
        np.save('biases',self.biases)
        np.save('weights',self.weights)

    def load(self):
        self.biases = ('biases.npy')
        self.weights = ('weights.npy')

    
        





    
def softmax(z):
    shift = z - np.max(z)
    exp_z = np.exp(shift)
    return exp_z/np.sum(exp_z)
    
def sigmoid_function(z):
    
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    
    return sigmoid_function(z)*(1-sigmoid_function(z))

def tanh_function(z):
    
    return 2.0/(1.0+np.exp(-2*z)) - 1

def ReLU_function(z):

    return np.where(z < 0, 0, z)

def softplus_function(z):

    return np.log(1 + np.exp(z))


