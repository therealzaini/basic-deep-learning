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

