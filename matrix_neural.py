import numpy as np
#import matplotlib.pyplot as plt
def sigmoid(z):
    return 1/(1+ np.exp(-z))

def sig_prime(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))




np.random.seed(3)

def feed(iterations = 20000):
    e = []
    target =np.random.randint(1,2,size = (4,4))
    print(target)
    bias = np.random.randn(4,4)
    weights = np.random.randn(4,4)
    data = np.random.randint(1,10, size = (4,4))

    for i in range(iterations):
        output = sigmoid(np.dot(data,weights)+bias)
        errors = (output - target)**2
        e.append(errors)

        z_w = data
        cost_p = 2 * (errors - target)
        sigmoid_p = sig_prime(output)
        c_w = np.dot(np.dot(z_w,sigmoid_p), cost_p)
        c_b = np.dot(np.dot(1,sigmoid_p), cost_p)
#        print(cost_p)

        weights = weights -(.001 * c_w)
        bias = bias -(.001 * c_b)
        for er, w,b in zip(errors,weights,bias):

                print("{}\t\t\t{}\t\t\t{}".format(er,w,b))

    return data,weights,bias,target



data,weights,bias,target=feed()
output = sigmoid(np.dot(data,weights)+bias)
print("Training complete.\nTarget:\t{}\nOutput:{}".format(target,output))
