Forward Propagation
-------------------

Definition:

.. code-block:: python

    forward_propagate(input_vector: Matrix): -> tuple[Matrix, tuple[list[Matrix], list[Matrix]]]

If ``input_matrix`` is not a column vector, a ``ValueError`` will be raised.

Let :math:`L` be the number of layers, :math:`\left(W^{(1)},\cdots,W^{(L-1)}\right)` be the 
:math:`L-1` weight matrices and :math:`\left(B^{(1)},\cdots,B^{(L-1)}\right)` be the 
:math:`L-1` bias matrices. We iteratively compute the output of the MLP 
based on the input vector :math:`A^{(0)}=X` via the recursive formula:

.. math::

    \forall n \in [1, L], \quad A^{(n)} = f\left(W^{(n)}\cdot A^{(n-1)} + B^{(n)}\right)

The method returns a tuple ``(output_vector, (activations, pre_activations))``.

``output_vector`` is self explanatory: the output of the MLP based on the input.