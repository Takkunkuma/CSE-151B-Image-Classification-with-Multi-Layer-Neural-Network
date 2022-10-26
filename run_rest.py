import os
args = [
    'test_regularization',
    'test_activation',
]

for test in args:
    print(f'now running {test}')
    print()
    os.system(f'python main.py --experiment {test}')
    print()
