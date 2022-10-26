import os
args = [
    'test_momentum',
    'test_regularization',
    'test_activation',
    'test_hidden_units',
    'test_hidden_layers',
    
]

for test in args:
    print(f'now running {test}')
    print()
    os.system(f'python main.py --experiment {test}')
    print()
