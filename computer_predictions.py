import numpy as np
import pygame
from time import sleep
pygame.mixer.init();
pygame.init();
#1 = Red
#0 = Blue

data = [[3,  1.5,  1],
        [2,  1,    0],
        [4,  1.5,  1],
        [3,  1,    0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5, 1,   1],
        [1,  1,    0]
        ]
red = pygame.mixer.Sound("red_voice.wav");
blue = pygame.mixer.Sound("blue_voice.wav");
intro = pygame.mixer.Sound("intro_voice.wav");
find_point = [5,5] #find its output





def sigmoid(z):
    return 1/(1 + np.exp(-z))



def training():
    w1 =np.random.randn() #Start off with random numbers
    w2 = np.random.randn()
    bias = np.random.randn()
    

    #Training loop
    for i in range(100000):
        index= np.random.randint(len(data)) #allows to retrieve any random point
        point = data[index]
        target = point[2]
        
        z = (point[0] * w1) + ( point[1] * w2) + bias
        prediction = sigmoid(z)

        slope = 2 * (prediction - target)


        dw1 = point[0]
        dw2 = point[1]
        dbias = 1

        cost_w1 = slope * dw1
        cost_w2 = slope * dw2
        cost_bias = slope * dbias


        w1 = w1 -.1 * cost_w1
        w2 = w2 -.1 * cost_w2
        bias = bias -.1 * cost_bias
        #print("{0}\t{1}\t{2}".format(w1,w2,bias))
    return w1, w2,bias

       
        
    



w1, w2,bias = training()

for i in range(len(data)):
    
    z = w1 * data[i][0] + w2 * data[i][1] + bias
    prediction = sigmoid(z)
    if prediction >= .5:
        
        red.play()
        
        print("This flower is RED ")
        print(prediction)

    else:
        blue.play()
        print("This flower is BLUE")
        print(prediction)
    sleep(3)

   




res = w1 * find_point[0] + w2 * find_point[1] + bias
print("\nFinding result for {}\n".format(find_point))
prediction = sigmoid(res)
if prediction >= .5:
    print("From the data you provided, I know that this flower RED")
    intro.play()
    sleep(2.8)
    red.play()
   
   

else:
    print("From the data you provided, I know that this flower Blue")
    intro.play()
    sleep(2.8)
    blue.play()
   
    







