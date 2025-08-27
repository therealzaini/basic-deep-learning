Getters and Setters
-------------------

For a given matrix :math:`A = \begin{pmatrix} a_{1,1} & a_{1,2} & \cdots & a_{1,p} \\ a_{2,1} & a_{2,2} & \cdots & a_{2,p} \\ \vdots & \vdots & \ddots & \vdots & a_{n,1} & a_{n,2} & \cdots & a_{n,p} \end{pmatrix}`,
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