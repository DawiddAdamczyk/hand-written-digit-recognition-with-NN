# hand-written-digit-recognition-with-NN-and-GA
Implemented from scratch neural network and evolutionary algorithm for recognition of hand written digits. Tested and trained against the MNIST dataset, which contains 60,000 training data examples and a test dataset of 10,000 elements. The images are stored in grayscale and have dimensions of 28x28.


Written as a part of "Artificial Intelligence" course taken at Gdańsk University of Technology, 2020.
# Status of project
This project has been completed. 
# Neural network
## Description of used method
A gradient algorithm with back propagation was used to build the network. The network consists of interconnected layers, starting from input layer (in case of MNIST vector size is 28x28=784), through hidden layers, ending on output layer (vector 10 giving the probability of guessing a given digit).
## Implementation of neural network
A fully connected neural network was trained. The function that computes the results on each neuron was taken as the
Sigmoid function: 1/(1 + 𝑒^-x). The error function (against which the derivatives are calculated) is the square of the error in the output. The training examples were given in random order. The starting values on the connections were initialized randomly from some interval.
