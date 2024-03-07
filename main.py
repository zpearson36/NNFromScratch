import activation_functions as af
import layer

import nnfs
from nnfs.datasets import spiral_data

if __name__ == '__main__':
    nnfs.init()

    X = [
          [1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
             ]

    X, y = spiral_data(100,3)
    a_function = af.ActivationFunction("rectifiedlinear")
    layer1 = layer.LayerDense(2, 5)
    layer1.forward(X)
    print(layer1.output)
    a_function.forward(layer1.output)
    print(a_function.output)
    
