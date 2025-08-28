Activation Functions Registry 
==============================

The module named ``ActivationFunctionRegistry`` contains
the most used activation functions, allowing easy access 
to them while making a neural network.

.. code-block:: python

    from basic_deep_learning import*
    from basic_deep_learning import ActivationFunctionsRegistry as afr

The registry contains for as the current version the following activation functions:

+------------------------+---------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+
| Function               | Definition                                                                                        | Deriviative                                                                          |
+========================+===================================================================================================+======================================================================================+
| Sigmoid                | :math:`\sigma(z) = \displaystyle\frac{1}{1+e^{-z}}`                                               | :math:`\sigma'(z)= \sigma(z)\left(1-\sigma(z)\right)`                                |
+------------------------+---------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+
| ReLU                   | :math:`\mathrm{ReLU}(z) = \max(0, z) = \begin{cases} z \quad & z\geq 0\\ 0 \quad & z<0\end{cases}`|:math:`\mathrm{ReLU}'(z) =\begin{cases} 1 \quad & z\geq 0 \\ 0 \quad & z<0\end{cases}`|
+------------------------+---------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+
|Linear                  |:math:`f(z)=z`                                                                                     | :math:`f'(z)=1`                                                                      |
+------------------------+---------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+
|Hyperbolic tangent      |:math:`f(z) = \tanh(z)`                                                                            |:math:`f'(z)=1-\tanh^2(z)`                                                            |
+------------------------+---------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+

Each function and its deriviative are then grouped in tuples of ``(activation_function, deriviative)``
and stored in a dictionnary named ``Activations`` where the keys are strings indicating the name of the function.

For example,

.. code-block:: python

    from basic_deep_learning import*
    from basic_deep_learning import ActivationFunctionsRegistry as afr

    sigmoid, sigmoid_prime = afr.Activations['sigmoid']

    print(sigmoid(1))
    print(sigmoid_prime(0))

.. code-block:: bash

    0.7310585786300049
    0.25

The keys for the activation functions are respectively ``'sigmoid'``, ``'ReLU'``,
``'linear'`` and ``'tanh'``.

These activation functions can take as parameters integers, floatiing-point values,
lists and ever ``Matrix`` instances; the return type is the same as the input
where it is computed component-wise for higher order data structures.

.. code-block:: python

    from basic_deep_learning import*
    from basic_deep_learning import ActivationFunctionsRegistry as afr

    tanh = afr.Activations['tanh'][0]

    z = 2
    v = [-1, 0, 1]
    M = Matrix([[1, 2, 3],[0, 1, -2]])

    print(tanh(z))
    print(tanh(v))
    print(tanh(M))

.. code-block:: bash

    0.9640275800758169
    [-0.7615941559557649, 0.0, 0.7615941559557649]
    matrix([[0.7615941559557649, 0.9640275800758169, 0.9950547536867305], [0.0, 0.7615941559557649, -0.9640275800758169]])

The registry contains as well the softmax function that turns a 
column vector into a probability distribution. More formally, 
if :math:`X = \begin{pmatrix}x_1\\x_2\\ \vdots\\x_n\end{pmatrix}`
is a column matrix, then :math:`\mathrm{softmax}(X) = \begin{pmatrix} y_1\\ y_2 \\ \vdots \\ y_n\end{pmatrix}`
where 
.. math::
   \forall i \in [1, n], \quad y_i = \frac{e^{x_i}}{\sum_{k=1}^n e^{x_k}}

For example,