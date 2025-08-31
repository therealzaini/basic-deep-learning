Additional Methods
------------------

Definition:

.. code-block:: python

    zero(n: int, p: int) -> Self

.. code-block:: python

    randomize(n: int, p: int, min_value: int|float, max_value: int|float) -> Self

The ``zero`` method takes in the number  ``n`` of rows and the number 
``p`` of columns and returns a ``Matrix`` instance of the format 
``(n, p)`` whose entries are all zeros.

.. code-block:: python

    from basic_deep_learning import*

    M = Matrix.zero(4,3)

    print(M)

.. code-block:: bash

    matrix([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
    ])

As for the ``randomize`` method, it takes in 
the number of rows and columns, as well as two numbers
``min_value`` and ``max_value`` to generate a matrix 
whose entries are randomly chosen in that interval.

.. code-block:: python

    from basic_deep_learning import*

    M = Matrix.randomize(2, 5, -2, 2)

    print(M)

.. code-block:: bash

    matrix([
            [1.5461630175389574, -1.7669737907829224, 0.7995896229747013, -0.4357429405356923, 0.4006793960718218],
            [-1.5718009270694329, 1.3578001519095744, 0.6738491556290129, -0.8341619836654566, -0.9217531463918105]
    ])

