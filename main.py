################################################################################
# CSE 151B: Programming Assignment 2
# Fall 2022
# Code by Chaitanya Animesh & Shreyas Anantha Ramaprasad
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import gradient
from constants import *
from train import *
from gradient import *
import argparse

# TODO


def main(args):

    # Read the required config
    # Create different config files for different experiments
    configFile = None  # Will contain the name of the config file to be loaded
    if (args.experiment == 'test_gradients'):  # 2b
        configFile = 'config_test.yaml'  # Create a config file for 2b and change None to the config file name
    elif (args.experiment == 'test_momentum'):  # 2c
        configFile = "config_2c_default.yaml"
    elif (args.experiment == 'test_regularization'):  # 2d
        configFile = "config_2d_l2_001.yaml"  # Create a config file for 2d and change None to the config file name
    elif (args.experiment == 'test_activation'):  # 2e
        configFile = 'config_2e_sigmoid.yaml'  # Create a config file for 2e and change None to the config file name
    elif (args.experiment == 'test_hidden_units'):  # 2f-i
        configFile = 'config_2f_dbl_hu.yaml'  # Create a config file for 2f-i and change None to the config file name
    elif (args.experiment == 'test_hidden_layers'):  # 2f-ii
        configFile = 'config_2f_2_layer.yaml'  # Create a config file for 2b and change None to the config file name

    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(
        path=datasetDir)  # Set datasetDir in constants.py

    # Load the configuration from the corresponding yaml file. Specify the file path and name
    # Set configYamlPath, configFile  in constants.py
    config = util.load_config(configYamlPath + configFile)

    if (args.experiment == 'test_gradients'):
        gradient.checkGradient(x_train, y_train, config)
        return 1

    # Create a Neural Network object which will be our model
    model = Neuralnetwork(config)

    model_params = (
        f'learning rate: {model.learning_rate}'
        f'\nl2: {model.l2}'
        f'\nregularization constant: {model.regularization}'
        f'\nmomentum: {model.momentum}'
        f'\nmomentum gamma: {model.momentum_gamma}'
    )

    print(configFile)
    print(model_params)

    # train the model. Use train.py's train method for this
    bestModel, train_losses, train_accs, valid_losses, valid_accs, epochs = train(
        model, x_train, y_train, x_valid, y_valid, config)

    # test the model. Use train.py's modelTest method for this
    print('testind data size', y_test.shape)
    test_loss, test_acc = modelTest(bestModel, x_test, y_test)

    # Print test accuracy and test loss
    print('Test Accuracy:', test_acc, ' Test Loss:', test_loss)



    # generate plots
    util.plots(train_losses, train_accs, valid_losses, valid_accs, epochs, configFile)


if __name__ == "__main__":

    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum',
                        help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)
