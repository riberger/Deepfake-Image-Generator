import numpy as np
from losses import *
from layers import *

class NeuralNet:
    """
    A class for learning with fully connected neural networks

    Parameters
    ----------
    d: int
        Dimensions of the input
    est_lossderiv: function ndarray(N) -> ndarray(N)
        Gradient of the loss function with respect to the inputs
        to the last layer, using the output of the last layer
    """
    def __init__(self, d, est_lossderiv):
        # Parameters for the network
        self.d = d
        self.Ws = []      #NOTE: L = len(self.Ws)
        self.bs = []
        self.fs = []
        self.est_lossderiv = est_lossderiv

        # Temporary variables to store outputs of each layer
        self.h = []
        self.a = []
        
    
    def add_layer(self, m, f, fderiv):
        """
        Parameters
        ----------
        m: int
            Number of nodes in the layer
        f: function ndarray(N) -> ndarray(N)
            Activation function, which is applied element-wise
        fderiv: function ndarray(N) -> ndarray(N)
            Derivative of activation function, which is applied element-wise
        """
        n = self.d
        if len(self.Ws) > 0:
            n = self.Ws[-1].shape[0]
        self.Ws.append(np.random.randn(m, n))
        self.bs.append(np.random.randn(m))
        self.fs.append({"f":f, "fderiv":fderiv})

    
    def forward(self, x):
        """
        Do a forward pass on the network
        
        Parameters
        ----------
        x: ndarray(d)
            Input to feed through
        
        Returns
        -------
        ndarray(m)
            Output of the network
        """
        self.h = [x]
        self.a = [None]
        for k in range(len(self.Ws)):
            a = self.bs[k] + self.Ws[k].dot(self.h[k])
            h = self.fs[k]["f"](a)
            self.a.append(a)
            self.h.append(h)
        return self.h[-1].flatten()

    def backprop_descent(self, x, y, alpha=0.01):
        """
        Do stochastic gradient backpropagation to compute the 
        gradient of all parameters on a single example, then 
        take a step against that direction by the learning rate
        
        Parameters
        ----------
        x: ndarray(d)
            Input to feed through
        y: float or ndarray(k)
            Goal output.  Dimensionality should match dimensionality
            of the last output layer
        alpha: float
            Learning rate
        """
        self.forward(x) # Always do the forward step first to compute the a's and h's
        ## TODO: Fill this in to complete backpropagation and weight updates
        
        L = len(self.Ws) #Number of Layers
        y_est = self.h[L]
       # a = self.a
        b_derivs = [None]*L
        g = self.est_lossderiv(y_est , y)
        W_derivs = [None]*L
        h = self.h
        f_derivs = 0
         
        
        #Step 1
        for k in range(L,0,-1):
            if k < L:   
                f_derivs = self.fs[k-1]["fderiv"]

                g = g * f_derivs(self.a[k])
                
        #Step 2
            b_derivs[k-1] = g               
            W_derivs[k-1] = np.outer(g, self.h[k-1])
            
        #Step 3
            g = self.Ws[k-1].T.dot(g)
            
        for i in range(len(W_derivs)):
            self.Ws[i] -= alpha * W_derivs[i]
            self.bs[i] -= alpha * b_derivs[i]
                              
        
        
      
        