Linear Algebra 
--------------

The package contains a ``LinearAlgebraUtils`` class providing 
some helper functions under the hood.

Definition:

.. code-block:: python 

    dot(u: list[int|float], v: list[int|float]) -> int|float

The function computes the dot product of the two lists interpreted as vectors.
If either one of the vectors passed is empty or if they have different lengths, 
the function will raise a ``ValueError``.

.. code-block:: python

    from basic_deep_learning import LinearAlgebraUtils as LAU
    from basic_deep_learning import*

    u = [3, 1, 1]
    v = [-1, 2, 1]

    print(f'u . v = {LAU.dot(u, v)}')

.. code-block:: bash

    u . v = 0

