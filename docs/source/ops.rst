Operations
----------

We are able to use a couple of operators to perform various operations on ``Matrix``
instances.

.. code-block:: python

    from basic_deep_learning import*

    A = Matrix([[1, 2, -4], [2, 0, 1]])
    B = Matrix([[6, -1, 3], [2, 1, 0]])

    print(f"A + B = {A + B}")  #Addition
    print(f"A - B = {A - B}")  #Substraction
    print(f"A @ B = {A @ B}")  #Component-wise multiplication
    print(f"3 * A = {3 * A}") #Scaling
    print(f"B / 4 = {B / 4}") #Component-wise division

.. code-block:: bash

    A + B = matrix([
            [7.0, 1.0, -1.0],
            [4.0, 1.0, 1.0]
    ])
    A - B = matrix([
            [-5.0, 3.0, -7.0],
            [0.0, -1.0, 1.0]
    ])
    A @ B = matrix([
            [6.0, -2.0, -12.0],
            [4.0, 0.0, 0.0]
    ])
    3 * A = matrix([
            [3.0, 6.0, -12.0],
            [6.0, 0.0, 3.0]
    ])
    B / 4 = matrix([
            [1.5, -0.25, 0.75],
            [0.5, 0.25, 0.0]
    ])

Performing an addition, substraction or component-wise multiplication
between two matrices who do not have the same format will raise a ``ValueError``.

Once the number of columns of the first matrix is equal to the number of rows 
of the second, we can perform matrix multiplication.

.. code-block:: python

    from basic_deep_learning import*

    A = Matrix([[1, 2, -4], [2, 0, 1]])
    B = Matrix([[-1, 0, 1, 2], [0, 1, 2, -1], [1, 2, -1, 0]])

    print(A * B)

.. code-block:: bash

    matrix([
            [-5.0, -6.0, 9.0, 0.0],
            [-1.0, 2.0, 1.0, 4.0]
    ])

But ``print(B*A)`` yields:

.. code-block:: bash

    TypeError: Invalid matrix formats (4≠2).


Definition:

.. code-block:: python

    T() -> Self

The ``T()`` method takes no arguments and simply returns the tranposed 
matrix.

.. code-block:: python

    print(A.T())

.. code-block:: bash

    matrix([
            [1, 2],
            [2, 0],
            [-4, 1]
    ])

Finally, we can use the ``==`` operator between ``Matrix`` 
instances; ``A == B`` is ``True`` if, and only if 
their ``matrix`` attributes are equal.
