Decorators
----------

The package includes a ``@extend_to_matrices``
decorator for function that takes in an integer or
floating-point value and returns an integer or floating-point
value. The decorator extends the function
for higher order data structures such as 
Python lists or ``Matrix`` instances, applying it 
component-wise.

.. code-block:: python

    import math
    from basic_deep_learning import*

    @extend_to_matrices
    def the_multi_dimentional_gamma_function(x):
        return math.gamma(x)

    x, y, z, t = 1/2, 2, 3, 4

    v = [x, y, z, t]

    M = Matrix([[x, y], [z, t]])

    print(f'Gamma(x) = {the_multi_dimentional_gamma_function(x)}')
    print(f'Gamma(v) = {the_multi_dimentional_gamma_function(v)}')
    print(f'Gamma(M) = {the_multi_dimentional_gamma_function(M)}')

.. code-block:: bash

    Gamma(x) = 1.7724538509055159
    Gamma(v) = [1.7724538509055159, 1.0, 2.0, 6.0]
    Gamma(M) = matrix([
            [1.7724538509055159, 1.0],
            [2.0, 6.0]
    ])

