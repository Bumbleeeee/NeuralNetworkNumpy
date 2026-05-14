## Implementation of Neural Network for MNIST image classification with NumPy only

Note that much of the code in data.py is taken directly from the NumPy tutorial for a similar project. 
The goal of this project was not to spend time learning how to download data.

The code and comments present in main.py serve to document my learning of how to implement forward and backward passes for a neural network.

Some key takeaways:

- Increasing the number of neurons in earlier layers has a greater effect on the accuracy than in later layers, likely due to the network
  being able to preserve more information for longer, and thus make more informed changes to the weights and biases.
  For example, increasing from 16 to 64 neurons in the first hidden layer increased the test accuracy from ~95% to ~97.5%, but in the second
  hidden layer (for a network with 2 hidden layers), the difference was negligible.
- Removing the second hidden layer has negligible impact on performance, but adding a third renders the network pretty much useless.
  I suspect this is a product of the specifics of this network, such as weight initializations and using sigmoid for nonlinearity, though I
  need to conduct further testing to pinpoint the exact problem.
