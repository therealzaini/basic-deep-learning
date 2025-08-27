Operations
----------

We are able to use a couple of operators to perform various operations on ``Matrix``
instances.

.. code-block:: python

    from basic_deep_learning import*

    A = Matrix([[1, 2, -4], [2, 0, 1]])
    B = Matrix([[6, -1, 3], [2, 1, 0]])

    print(A+B)  #Addition
    print(A-B)  #Substraction
    print(A@B)  #Component-wise multiplication
    print(3*A)  #Scaling

.. code-block:: bash

    matrix([[7, 1, -1], [4, 1, 1]])
    matrix([[-5, 3, -7], [0, -1, 1]])
    matrix([[6, -2, -12], [4, 0, 0]])
    matrix([[3, 6, -12], [6, 0, 3]])

Performing an addition, substraction or component-wise multiplication
between two matrices who do not have the same format will raise a ``ValueError``.

Once the number of columns of the first matrix is equal to the number of rows 
of the second, we can perform matrix multiplication.

.. code-block:: python

    from basic_deep_learning import*

    A = Matrix([[1, 2, -4], [2, 0, 1]])
    B = Matrix([[-1, 0, 1, 2], [0, 1, 2, -1], [1, 2, -1, 0]])

    print(A*B)

.. code-block:: bash

    matrix([[-5, -6, 9, 0], [-1, 2, 1, 4]])

But ``print(B*A)`` yields:

.. code-block:: bash

    ValueError: Invalid matrix formats (4≠2).

The ``T()`` method takes no arguments and simply returns the tranposed 
matrix.

.. code-block:: python

    print(A.T())

.. code-block:: bash

    matrix([[1, 2], [2, 0], [-4, 1]])

Finally, we can use the ``==`` operator between ``Matrix`` 
instances; ``A == B`` is ``True`` if, and only if 
their ``matrix`` attributes are equal.
