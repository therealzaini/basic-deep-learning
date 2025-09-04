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

The class constructor has three parameters: ``structure`` is the argument to which we pass 
a list whose elements represent the number of neurons in each layer.
For example, if we wish to make a neural network with 16 input neurons,
two hidden layers with each 32 neurons, and an output layer with 10 neurons, we would pass 
``[16, 32, 32, 10]`` to the ``structure`` parameter. The value passed will be hence be the ``self.structure``
attribute for the class.

The second and thrid parameters are ``hidden_layer_activation_function_label`` 
and ``output_layer_activation_function_label``
and each should be a string and one of the keys from the ``ActivationFunctionsRegistry.Activations``
`dictionnary <https://basic-deep-learning.readthedocs.io/en/main/activations_registry.html>`_.
As their names indicate, they are the respective activation functions for the hidden layers and output layer.

A class instance would have a couple other attributes:

``self.number_of_layers``: self explanatory.

``self.weights``: a list of the weight matrices as instances from the ``Matrix`` class, whose entries are
randomized (between -1 and 1 according to a uniform distribution).

``self.biases``: analogous to the previous (the matrices are column vectors).

``self.hidden_activation_function`` and ``self.hidden_activation_function_prime``: respectively
the hidden layers activation function and its deriviative.

``self.output_activation_function`` and ``self.output_activation_function_prime``: respectively
the output layer activation function and its deriviative.


.. code-block:: python

    from basic_deep_learning import*
    from basic_deep_learning import MultiLayerPerceptron as MLP

    nn = MLP([2, 3, 4, 2], 'ReLU', 'tanh')

    Ws = nn.weights
    Bs = nn.biases

    for i, W in enumerate(Ws):
        print(f'Weight matrix {i+1} : format {W.format} :\n{W}\n')

    for i, B in enumerate(Bs):
        print(f'Bias matrix {i+1} : format {B.format} :\n{B}\n')

.. code-block:: bash

    Weight matrix 1 : format (3, 2) :
    matrix([
            [0.33687682292717147, 0.46488420046410095],
            [-0.4798362744575957, 0.6349819224340718],
            [-0.9026429378248222, -0.7205940050419231]
    ])

    Weight matrix 2 : format (4, 3) :
    matrix([
            [-0.9505586648121618, -0.7962311792480947, 0.8638984951478819],
            [0.6571873101914014, 0.9860307652895879, 0.5286199836295016],
            [-0.5654737447575844, 0.06834237964549339, 0.8373605648182096],
            [-0.26281651046117727, -0.4316426593678664, -0.16536461024214333]
    ])

    Weight matrix 3 : format (2, 4) :
    matrix([
            [0.4533438769816358, -0.22660208605642973, 0.08829164849253113, -0.17251410257294686],
            [-0.0002389516444403217, 0.2234508414248606, 0.09136879884015414, 0.972088215631062]
    ])

    Bias matrix 1 : format (3, 1) :
    matrix([
            [0.494641188762706],
            [-0.20371517540725392],
            [-0.3833049509025652]
    ])

    Bias matrix 2 : format (4, 1) :
    matrix([
            [-0.8380501475592002],
            [0.03739240084104223],
            [-0.6114256685276511],
            [-0.609764017532284]
    ])

    Bias matrix 3 : format (2, 1) :
    matrix([
            [0.9583723688793484],
            [0.5947527143415694]
    ])

