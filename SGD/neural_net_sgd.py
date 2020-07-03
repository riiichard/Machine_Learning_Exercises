def fit_SGD(self, X, y):
        ##################### same as fit #########################
        if y.ndim == 1:
            y = y[:,None]
            
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1]>1 # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes)-1):
            W = scale * np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i])
            b = scale * np.random.randn(1,self.layer_sizes[i+1])
            weights.append((W,b))
        weights_flat = flatten_weights(weights)
        ##################### same as fit #########################
        
        # Apply SGD
        # Use a minibatch size of 500 and a constant learning rate of 0.001
        
        minibatch = 500
        rate = 0.001
        n_datapoints = X.shape[0]
        n_batchs = n_datapoints // minibatch
        # 1. For each Epoch inside the fit function
        for epoch in range(self.max_iter):
            for one_batch in range(n_batchs):
                # 2. Make a batch of size = 500 of your training data 
                #    using np.random.choice function in Python.
                #    batch is the indices of selected data
                batch = np.random.choice(n_datapoints,size=minibatch,replace=False)
                # 3. Use the given objective function (def funObj) inside the Neural Network class 
                #    to compute the derivative of objective function via weights and batch.
                f, g = self.funObj(weights_flat, X[batch], y[batch])
                # 4. Compute new weights using the function (update weights_flat)
                weights_flat = weights_flat - rate * g
        # 5. Unflatten weights using the given helper function (just as before)
        self.weights = unflatten_weights(weights_flat, self.layer_sizes)
