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

    matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

As for the ``randomize`` method, it takes in 
the number of rows and columns, as well as two numbers
``min_value`` and ``max_value`` to generate a matrix 
whose entries are randomly chosen in that interval.

.. code-block:: python

    from basic_deep_learning import*

    M = Matrix.randomize(2, 5, -2, 2)

    print(M)

.. code-block:: bash

    matrix([[-1.3161797158481972, -1.6078024493802117, -1.7801098541400786, 0.9349200301944229, 0.5681831961720363],
            [-1.0979995587900078, 1.03500040714135, 1.141746111182127, 0.3218842404651907, 0.730709706747739]])

