Initialisation
---------------

The ``MultiLayerPerceptron`` class is the main class that encapsulates a classic neural network.

Definition:

.. code-block:: python

    class MultiLayerPerceptron(
        structure: list[int],
        hidden_layer_activation_function_label: Literal['sigmoid', 'ReLU', 'linear', 'tanh'],
        output_layer_activation_function_label: Literal['sigmoid', 'ReLU', 'linear', 'tanh']
    )

The class constructor has three arguments: ``structure`` is the argument to which we pass 
a list whose elements represent the number of neurons in each layer.
For example, if we wish to make a neural network with 16 input neurons,
two hidden layers with each 32 neurons, and an output layer with 10 neurons, we would pass 
``[16, 32, 32, 10]`` to the ``structure`` parameter.