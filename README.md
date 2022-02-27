# hand-written-digit-recognition-with-NN
Implemented from scratch neural network for recognition of hand written digits. Tested and trained against the MNIST dataset, which contains 60,000 training data examples and a test dataset of 10,000 elements. The images are stored in grayscale and have dimensions of 28x28.


Written as a part of "Artificial Intelligence" course taken at Gdańsk University of Technology, 2020.
# Status of project
This project has been completed. 
# Neural network
## Description of used method
A gradient algorithm with back propagation was used to build the network. The network consists of interconnected layers, starting from input layer (in case of MNIST vector size is 28x28=784), through hidden layers, ending on output layer (vector 10 giving the probability of guessing a given digit).
## Implementation of neural network
A fully connected neural network was trained. The function that computes the results on each neuron was taken as the
Sigmoid function: 1/(1 + 𝑒^-x). The error function (against which the derivatives are calculated) is the square of the error in the output. The training examples were given in random order. The starting values on the interconnections were initialized randomly from some interval.
## Single-threaded vs mulit-threaded computing
The calculations were performed on a single core, optimized using @vectorize and @guvectorize decorators from the Numba python library (the speedup is significant). In the AI class responsible for the neural network mechanism, it is possible to choose (variable target) whether the computation of functions performing operations on vectors and matrices should be performed on a single or multiple cores or a graphics card ('cpuparallel','','cuda'), but the most efficient computations are on a single core (probably the cost associated with data transfer between graphics cards, also the multi-core CPU turns out to be faster only for operations on very large vectors).
In order to run the program faster (load examples faster), the example sets are cached to a file as ready-made python objects.
## Results
The best result achieved was about 98.2% on the test set (a model containing successively 10, 500, 500, 784 neurons, with slight overfitting, a lot of epochs and a fairly small "learning rate"). The network was also tested using drawn digits in Paint, where it recognized the vast majority of digits, but with slightly worse results.
