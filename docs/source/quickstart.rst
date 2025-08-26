Quickstart
==================

To make sure the pacakge is properly install,
we can start by performing a simple operation,
such as matrix multiplication:

.. code-block:: python
    
    from basic_deep_learning import*

    A = Matrix([[-1,1,0,3],[1,3,-1,2],[0,1,0,-1]])
    B = Matrix([[3,-1],[-4,2],[2,1],[-1,0]])

    print(A*B)