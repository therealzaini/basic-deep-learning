Initialisation
--------------

We pass to the ``Matrix`` class constructor a list of rows (as lists) of a given matrix.
For example, the matrix :math:`M = \begin{pmatrix} 1 & 0 \\ 0 & 1\end{pmatrix}`
will be implemented as such:

.. code-block:: python

    from basic_deep_learning import*

    M = Matrix([[1, 0], [0, 1]])

Use ``print()`` to display the matrix.

.. code-block:: python

    from basic_deep_learning import*

    M = Matrix([[1, 0], [0, 1]])

    print(M)

.. code-block:: bash

    matrix([[1, 0], [0, 1]])

Each ``Matrix`` instance has two attributes; ``format`` and ``matrix``.
The ``format`` is a tuple of two elements, the first being the number of rows and the latter 
being the number of columns.

.. code-block:: python

    from basic_deep_learning import*

    A = Matrix(
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    )

    print(A.matrix)
    print(A.format)

.. code-block:: bash

    [[1, 2, 3], [4, 5, 6]]
    (2, 3)
