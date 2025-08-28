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