Matrix
------


The fundamental package that implements matrices.

We pass to the ``Matrix`` class constructor a list of rows (as lists) of a given matrix.
For example, the matrix :math:`M = \begin{pmatrix} 1 & 0 \\ 0 & 1\end{pmatrix}`
will be implemented as such:

.. code-block:: python

    from basic_deep_learning import*

    M = Matrix([[1, 0], [0, 1]])

