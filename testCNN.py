from MultiLayerPerceptron import *
from CNN import *
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
training, validation, test = mnist_loader.load_data()
offset = 5000
t = training[0][0:offset],training[1][0:offset]
offsetv = 500
v = validation[0][0:offsetv],validation[1][0:offsetv]
epochs = 30





reps = 5
opti_names = ['K=3','K=9','K=11','K=13'] 
x_list = []
y_list = []
for rep in range(0,reps):
    
    a = CNN([6272,100,10],None,3,1,1)
    _,avg_x,avg_y,_ = a.train(t,v,epochs,0.05,64,softplus_function,sigmoid_function,20)
    avg_x = [element / reps for element in avg_x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in avg_y]
    y_list.append(avg_y)

    a = CNN([6272,100,10],None,9,1,4)
    _,avg_x,avg_y,_ = a.train(t,v,epochs,0.05,64,softplus_function,sigmoid_function,20)
    avg_x = [element / reps for element in avg_x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in avg_y]
    y_list.append(avg_y)

    a = CNN([6272,100,10],None,11,1,5)
    _,avg_x,avg_y,_ = a.train(t,v,epochs,0.05,64,softplus_function,sigmoid_function,20)
    avg_x = [element / reps for element in avg_x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in avg_y]
    y_list.append(avg_y)
    
    a = CNN([6272,100,10],None,13,1,6)
    _,avg_x,avg_y,_ = a.train(t,v,epochs,0.05,64,softplus_function,sigmoid_function,20)
    avg_x = [element / reps for element in avg_x]
    x_list.append(avg_x)
    avg_y = [element / reps for element in avg_y]
    y_list.append(avg_y)
    
    
    
    


 
plt.title('Skuteczność sieci w kolejnych epokach dla różnego rozmiaru filtra K')
plt.xlabel('epoki')
plt.ylabel('skuteczność')
for i in range(0,4):
    
    plt.plot(x_list[i],y_list[i],label=opti_names[i])
plt.legend()
plt.show()
plt.savefig('cnn_filter_size_summary.png')
plt.cla()