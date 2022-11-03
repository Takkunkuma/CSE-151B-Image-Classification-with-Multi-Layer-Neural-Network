## Sample running method
    python main.py --experiment test_activation 
## There is 1 optional arguments to run main
--experiment [str]
    controls with experiment to run with the given string, the values may be:
    - test_gradients
    - test_momentum [default]
    - test_regularization
    - test_activation
    - test_hidden_units
    - test_hidden_layers


## Config Files
One may change the architecture and hyperparameters of the model through changing the config file ran when specifying which experiment to run. The config files are located in the configs directory. The arguments listed in the config files are NOT OPTIONAL, meaning all of them have to be in the config files. The descriptions of the config files are in the yaml files. 


## Outputs
The code will return the average train loss, train accuracy, average validation loss, validation accuracy per epoch and also test the model on the test set after training and output the test accuracy along with the test loss. 

## Files created
Running the program will save 2 pngs in the plots directory, corresponding to the train and validation loss results along with the train and validation accuracy results. There is also accuracy.eps and loss.eps which both refers to the plot datas by in another format. Furthermore, there are 4 csv's saved, each corresponding to the train accuracy, train loss, validation accuracy, and validation loss values. These are all in the plots directory