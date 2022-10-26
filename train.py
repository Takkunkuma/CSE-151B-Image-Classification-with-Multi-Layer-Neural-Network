
import copy
from neuralnet import *

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the train set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """

    # Read in the esssential configs
    X_train = util.append_bias(x_train)
    X_valid = util.append_bias(x_valid)


    M = config['epochs']
    N = config['batch_size']
    K = config['early_stop_epoch']

    # for each epoch
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    fill_length = len(str(M))


    early_stop_epoch = 0

    for epoch in range(M):
        # train using minibatches
        for train_batch_X,train_batch_y in util.generate_minibatches((X_train, y_train), N):
            model.forward(train_batch_X, train_batch_y)
            model.backward(gradReqd=True)
        

        # calculate losses and store them
        curr_train_loss, curr_train_acc = model.forward(X_train, y_train)
        curr_valid_loss, curr_valid_acc = model.forward(X_valid, y_valid)

        # print result
        debug_msg = (
            f'epoch {str(epoch+1).zfill(fill_length)}, train loss {curr_train_loss:.6f}, valid loss {curr_valid_loss:.6f}, '
            f'train acc {curr_train_acc:.3f}, valid acc {curr_valid_acc:.3f}'
        )

    
        print(debug_msg)




        train_losses.append(curr_train_loss)
        train_accs.append(curr_train_acc)
        valid_losses.append(curr_valid_loss)
        valid_accs.append(curr_valid_acc)

        # save best model?

        # if early stopping
        if config['early_stop']:
            if len(valid_losses) >= K and (np.diff([valid_losses[-K:]]) >= 0).all():
                print('early stopping')
                early_stop_epoch = epoch - K
                break 
            

    return model, train_losses, train_accs, valid_losses, valid_accs, early_stop_epoch

#This is the test method
def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    
    return model.forward(util.append_bias(X_test), y_test)


