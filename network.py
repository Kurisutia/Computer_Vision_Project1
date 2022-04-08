class Network(object):
    '''
    Represents a neural network with any combination of layers
    '''
    def __init__(self, learning_rate,lamda):
        self.lr = learning_rate
        self.lamda=lamda
        self.layers = []
        self.loss = None
        self.step=2

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss
    
    def predict(self, inputs, train=False):
        scores = inputs
        #print(scores.shape)
        for layer in self.layers:
            scores = layer.forward(scores, train=train)
        return scores
    
    def train(self, inputs, labels,epoch):
        vars_and_grads = []
        n=len(labels)
        # Forward pass
        scores = self.predict(inputs, train=True)
        # Backward pass
        W=self.layers[-1].W
        loss, grad = self.loss.get_loss(scores, labels,self)
        for layer in reversed(self.layers):
            grad, layer_var_grad = layer.backward(grad)
            vars_and_grads += layer_var_grad
        
        if epoch>=self.step:
            self.lr=self.lr*0.2
            self.step=self.step*2

        # Gradient descent update:
        for var_grad in vars_and_grads:
            var, grad = var_grad
            if(var.shape[0]!=1):
                var -= self.lr *( grad+self.lamda/n*var)
            else:
                var-=self.lr * grad
        
        return loss