import numpy as np
from time import sleep
class nn:
    def __init__(self):
        self.weight_one = np.random.randn()
        self.weight_two = np.random.randn()
        self.bias = np.random.randn()


    def slope(self,prediction,target):
        cost=2*(prediction - target)
        return cost

    def sigmoid(self,prediction, target):
        prediction = 1/(1 + np.exp(-prediction)) #prediction probability of sigmoid
        
       # sleep(2)
        for i in range(200):
            prediction = prediction - .1 * self.slope(prediction, target)
            #print(prediction)
       # print("Done.\n")
        
        return prediction

    def neuron(self, m1,m2, target):
       # print("Sending: {0}, {1} Target {2}".format(m1,m2,target))
       # sleep(2)
        z = (m1 * self.weight_one) + (m2 * self.weight_two) + self.bias
        sigmoid_res = self.sigmoid(z, target)
        return sigmoid_res
