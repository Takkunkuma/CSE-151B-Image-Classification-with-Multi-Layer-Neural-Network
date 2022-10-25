import numpy as np
import util

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU","output"]:   #output can be used for the final layer. Feel free to use/remove it
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        TODO
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, a):
        """
        TODO: Implement the sigmoid activation here.
        """

        return 1 / (1 + np.exp(a))

    def tanh(self, a):
        """
        TODO: Implement tanh here.
        """

        return np.tanh(a)

    def ReLU(self, a):
        """
        TODO: Implement ReLU here.
        """
        return np.max(0, a)

    def output(self, a):
        """
        TODO: Implement softmax function here.
        Remember to take care of the overflow condition.
        """
        exp = np.exp(a)
        row_sum = exp.sum(axis=1)


        return (exp.T / row_sum).T
       

    def grad_sigmoid(self, x):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def grad_tanh(self, x):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1 - (self.tanh(x)**2)

    def grad_ReLU(self, x):
        """
        TODO: Compute the gradient for ReLU here.
        """
        return np.greater(x, 0).astype(int)

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """

        return np.ones(x.shape)


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation, weightType):
        """
        Define the architecture and create placeholders.
        """
        np.random.seed(42)

        self.w = None
        if (weightType == 'random'):
            self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None    # Save the input to forward in this
        self.a = None    #output without activation
        self.z = None    # Output After Activation
        self.activation = activation   #Activation function


        self.dw = 0  # Save the gradient w.r.t w in this. You can have bias in w itself or uncomment the next line and handle it separately
        # self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """

        self.a = self.x @ self.w
        self.z = self.activation(self.a)

        return self.z

    def backward(self, deltaCur=None, learning_rate=0.001, momentum_gamma=None, regularization=None, gradReqd=True):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        The delta expression (that you prove in PA1 part1) for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass
        """

        # delta_j
        delta_j = deltaCur @ self.activation.backward(self.a)

        self.dw = delta_j @ self.z

        # calculate delta_j & w_ij for all j
        to_return = delta_j @ self.w


        # update weights
        if gradReqd:
            self.w = self.w + learning_rate * self.dw

        return to_return
        


class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """

    def __init__(self, config):
        """
        Create the Neural Network using config. Feel free to add variables here as per need basis
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None        # For saving the output vector of the model
        self.targets = None  # For saving the targets

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation']),
                          config["weight_type"]))
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"]))

        self.learning_rate = config['learning_rate']

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)


    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        output = x
        # Compute forward pass through all the layers
        for i in range(self.num_layers):
            output = self.layers[i].forward(output)


        # output is now N X 10

        # return the acc
        if (targets is None):
            return output
        
        # if targets is given return 
        return output, util.calculateCorrect(output, targets)



    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''






        raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self, gradReqd=True):
        '''
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
        # don't update weights, only calculate delta backwards
        if (gradReqd == False):

            # backprop through all layers
            for i in range(self.num_layers - 1, -1, -1):
                # perfrom backprop and save delta for the previous layer
                if i == self.num_layers - 1:
                    # output layer: t-y
                    deltaPrev = self.layers[i].backward(
                        deltaCur=self.target - self.y, learning_rate=self.learningRate, gradReqd=False)
                elif i > 0:
                    # hidden layers
                    deltaPrev = self.layers[i].backward(
                        deltaCur=deltaPrev, learning_rate=self.learningRate, gradReqd=False)
                elif i == 0:
                    # not saving delta for input layer
                    self.layers[i].backward(
                        deltaCur=deltaPrev, learning_rate=self.learningRate, gradReqd=False)
        else:
            # update wieghts
            for i in range(self.num_layers - 1, -1, -1):
                self.layers[i].backward(
                    learning_rate=self.learningRate, gradReqd=True)
    





