from neural_network import nn
import numpy as np
network = nn()
res =[]

inputs = np.matrix('''
             3.5,1.2,1;
             1.4, 4, 0;
             2,3.5, 1;
             4,2.5,1;
             5.5,1,0;
             9,8,0;
             3.2,4,1;
             9.3,3.3,1;
             1.1,4.3,0;
             4.6,7,1;
             4,3.2,1;
             1,2,0;
             1.1,1,0
          ''')


print("Training in process...\n")
for i in range(len(inputs)):
    m1= inputs.item(i,0)
    m2 = inputs.item(i,1)
    target = inputs.item(i,2)
    my_neuron=network.neuron(m1,m2,target)
    #print("Output Result: {}".format(my_neuron))
    res.append(my_neuron)

print("Trained Values:\n")
for i in res:
    print(i)







































##import numpy as np
##
##weight_one = np.random.randn()
##weight_two = np.random.randn()
##bias = np.random.randn()
## 
##inputs = np.matrix('''
##             3.5,1.2,20;
##             1.4, 4, 2
##          ''')
##
##
##def slope(prediction,target):
##    cost=2 * (prediction - target)
##    return cost
##
##def sigmoid(prediction,target):
##    prediction = 1/(1 + np.exp(-prediction)) #prediction probability of sigmoid
##    print("Prediction: {0} Target {1}".format(prediction, target))
##    for i in range(200):
##        prediction = prediction - .1 * slope(prediction, target)
##        
##    print("\nDone.")
##    return prediction
##       
##        
##    
##
##def neural_network(target,input_one, input_two, w1,w2,b):
##    z = (input_one * w1) + (input_two * w2) + b
##    sigmoid_res = sigmoid(z,target)
##    return sigmoid_res
##
##
##
##
##
##
##
##input_one= inputs.item(0,0)
##input_two= inputs.item(0,1)
##target = inputs.item(0,2)
##
##nuron_result = neural_network(target,input_one,input_two, weight_one,weight_two,bias)
##print("Output from a Single Nuran --> {}".format(nuron_result))
