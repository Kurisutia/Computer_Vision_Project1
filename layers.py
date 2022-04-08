import numpy as np

class Layer(object):
    '''
    Abstract class representing a neural network layer
    '''
    def forward(self, X, train=True):
        '''
        Calculates a forward pass through the layer.
        '''
        raise NotImplementedError('This is an abstract class')

    def backward(self, dY):
        '''
        Calculates a backward pass through the layer.

        Args:
            dY (numpy.ndarray): The gradient of the output with dimensions (batch_size, output_size)

        Returns:
            dX, var_grad_list
            dX (numpy.ndarray): Gradient of the input (batch_size, output_size)
            var_grad_list (list): List of tuples in the form (variable_pointer, variable_grad)
                where variable_pointer and variable_grad are the pointer to an internal
                variable of the layer and the corresponding gradient of the variable
        '''
        raise NotImplementedError('This is an abstract class')

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        '''
        Represent a linear transformation Y = X*W + b
            X is an numpy.ndarray with shape (batch_size, input_dim)
            W is a trainable matrix with dimensions (input_dim, output_dim)
            b is a bias with dimensions (1, output_dim)
            Y is an numpy.ndarray with shape (batch_size, output_dim)

        W is initialized with Xavier-He initialization
        b is initialized to zero
        '''
        self.func="linear"
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0/input_dim)
        self.b = np.zeros((1, output_dim))

        self.cache_in = None

    def forward(self, X, train=True):
        out = np.matmul(X, self.W) + self.b
        if train:
            self.cache_in = X
        return out

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        db = np.sum(dY, axis=0, keepdims=True)
        dW = np.matmul(self.cache_in.T, dY)
        dX = np.matmul(dY, self.W.T)
        return dX, [(self.W, dW), (self.b, db)]

class relu(Layer):
    def __init__(self):
        self.func="relu"
        self.cache_in = None

    def forward(self, X, train=True):
        if train:
            self.cache_in = X
        return np.maximum(X, 0)

    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY * (self.cache_in >= 0), []

class tanh(Layer):
    def _init_(self):
        self.func="tanh"
        self.cache_in=None
        
    def forward(self,X,train=True):
        if train:
            self.cache_in=X
        return np.tanh(X)
    
    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY * (1 - np.tanh(self.cache_in) * np.tanh(self.cache_in)), []

class sigmoid(Layer):
    def _init_(self):
        self.func="sigmoid"
        self.cache_in=None
        
    def forward(self,X,train=True):
        if train:
            self.cache_in=X
        return (1/(1+np.exp(X*-1)))
    
    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        return dY * (np.exp(self.cache_in*-1)/(1+np.exp(self.cache_in*-1))**2), []

class leakyrelu(Layer):
    def _init_(self):
        self.func="leakyrelu"
        self.cache_in=None
    
    def forward(self,X,train=True):
        if train:
            self.cache_in=X
        return  np.maximum(0.1*X, X)
    
    def backward(self, dY):
        if self.cache_in is None:
            raise RuntimeError('Gradient cache not defined. When training the train argument must be set to true in the forward pass.')
        a=[]
        for i in range(self.cache_in.shape[0]):
            b=[]
            for j in range(self.cache_in.shape[1]):
                if self.cache_in[i][j]>0:
                    p=1
                else:
                    p=0.1
                b.append(p)
            a.append(b)
        #a=max(,a)
        return dY * a, []

class Loss(object):
    '''
    Abstract class representing a loss function
    '''
    def get_loss(self):
        raise NotImplementedError('This is an abstract class')

class SoftmaxCrossEntropyLoss(Loss):
    '''
    Represents the categorical softmax cross entropy loss
    '''

    def get_loss(self, scores, labels,network):
        '''
        Calculates the average categorical softmax cross entropy loss.

        Args:
            scores (numpy.ndarray): Unnormalized logit class scores. Shape (batch_size, num_classes)
            labels (numpy.ndarray): True labels represented as ints (eg. 2 represents the third class). Shape (batch_size)

        Returns:
            loss, grad
            loss (float): The average cross entropy between labels and the softmax normalization of scores
            grad (numpy.ndarray): Gradient for scores with respect to the loss. Shape (batch_size, num_classes)
        '''
        scores_norm = scores - np.max(scores, axis=1, keepdims=True)
        #print(scores)
        scores_norm = np.exp(scores_norm)
        scores_norm = scores_norm / np.sum(scores_norm, axis=1, keepdims=True)
        true_class_scores = scores_norm[np.arange(len(labels)),labels]
        #print(true_class_scores.shape)
        loss = np.mean(-np.log(true_class_scores))
        # print(loss)
        n=len(labels)
        L=0
        for layer in network.layers:
            if layer.func=="linear":
              W=layer.W
              W=W**2
              W_a=np.sum(W)
              L+=network.lamda*W_a/(2*n)
        loss=loss+L

        one_hot = np.zeros(scores.shape)
        one_hot[np.arange(len(labels)), labels] = 1.0
        grad = (scores_norm - one_hot) / len(labels)
        
        return loss, grad    

class SquareLoss(Loss):
    def get_loss(self,scores,labels):
        loss=0
        grad=[]
        for i in range(scores.shape[0]):
            k=scores[i]-labels[i]
            grad.append(k)
            loss+=0.5*k**2
        grad=np.array(grad)
        return loss,grad            
            