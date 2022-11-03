import numpy as np
from neuralnet import Neuralnetwork

def check_grad(model, x_train, y_train):

    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """

    #initialize epsilon
    epsilon = 10 ** (-2)

    #initialize array to store indexes pair
    indexes = []
    mainGradient = []
    numericalApprox = []
    absDiff = []
    weightTypes = ["input to hidden", "input to hidden", "bias to hidden",
                    "hidden to output", "hidden to output", "bias to output"]
    layerindex = [0, 0, 0, 1, 1, 1]     #2 input to hidden, bias to hidden, 2 hidden to output, bias to output
    counter = 0
    curLayer = 0

    #input to hidden * 2
    for i in range(2):
        itoh = []
        itoh.append(np.random.randint(1, 3073))
        itoh.append(np.random.randint(1, 129))
        indexes.append(itoh)

    # hidden bias
    indexes.append([0, np.random.randint(1, 129)])
    # output bias
    indexes.append([0, np.random.randint(0, 10)])

    #hidden to output * 2
    for i in range(2):
        htoo = []
        htoo.append(np.random.randint(1, 129))
        htoo.append(np.random.randint(0, 10))
        indexes.append(htoo)

    #make deepcopy of weights
    for i in indexes:
        copy_main = copy.deepcopy(model)
        copy_increment = copy.deepcopy(model)
        copy_decrement = copy.deepcopy(model)

        #counter for layerindex
        if counter == 3:
            curLayer += 1

        counter += 1

        #choosing a weight
        in_index = i[0]
        out_index = i[1]

        #weight choice of input 3rd unit to hidden 4th unit and update on deepcopy
        copy_increment.layers[layerindex[curLayer]].w[in_index, out_index] = \
                copy_increment.layers[layerindex[curLayer]].w[in_index, out_index] + epsilon
        copy_decrement.layers[layerindex[curLayer]].w[in_index, out_index] = \
                copy_decrement.layers[layerindex[curLayer]].w[in_index, out_index] - epsilon

        #forwardpass and backward pass the model with x_train_sample and y_train_sample
        x_train_bias = util.append_bias(x_train)
        copy_main.forward(x_train_bias, y_train)
        copy_main.backward(gradReqd=True)

        # Retrieve gradient from layer 0
        gradient = (-1) * copy_main.layers[layerindex[curLayer]].dw[in_index, out_index]
        #print("gradient: ", gradient)
        mainGradient.append(gradient)

        loss_decremented, correct_decrement = copy_decrement.forward(x_train_bias, y_train)
        loss_incremented, correct_increment = copy_increment.forward(x_train_bias, y_train)

        #Calculate Numerical Approximation
        approx_gradient = (loss_incremented - loss_decremented) / (2 * epsilon)
        numericalApprox.append(approx_gradient)
        absDiff.append(abs(approx_gradient - gradient))

    print(weightTypes)
    print("original: ", mainGradient)
    print("numerical: ", numericalApprox)
    print(absDiff)



def checkGradient(x_train,y_train,config):

    subsetSize = 10  #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)