Getters and Setters
-------------------

Definition:

.. code-block:: python

    get_entry(i: int, j:int) -> int|float

.. code-block:: python

    get_row(i: int) -> list[int|float]

.. code-block:: python

    get_column(j:int) -> list[int|float]

.. code-block:: python

    set_entry(value: int|float, i: int, j:int) -> None

For a given matrix :math:`A = \begin{pmatrix} a_{1,1} & a_{1,2} & \cdots & a_{1,p} \\ a_{2,1} & a_{2,2} & \cdots & a_{2,p} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n,1} & a_{n,2} & \cdots & a_{n,p} \end{pmatrix}`,
we access the entry :math:`a_{i,j}` using the ``get_entry(i, j)`` method.
The indexation of the entries is according to the mathematical convention, *ie*
starts at 1.

.. code-block:: python

    from basic_deep_learning import*

    A = Matrix(
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    )

    print(A.get_entry(1,3))

.. code-block:: bash

    3

We can also retrieve a row or column as a list:

.. code-block:: python

    print(A.get_row(2))
    print(A.get_column(1))

.. code-block:: bash

    [4, 5, 6]
    [1, 4]

To modify a certain entry from the matrix, we can use the
``set_entry(value, i, j)`` method:


.. code-block:: python

    A.set_entry(-1, 2, 2)   #Set the a_{2,2} entry to -1
    print(A)

.. code-block:: bash

    matrix([[1, 2, 3], [4, -1, 6]])

