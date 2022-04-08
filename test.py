import numpy as np
import pickle
import tensorflow as tf
from layers import Linear, relu, SoftmaxCrossEntropyLoss,tanh,sigmoid,leakyrelu
from network import Network
from main import load_data,validate_network

inputs, labels =  load_data()
with open("net",'rb') as f:
    net=pickle.load(f)
test_loss, test_acc = validate_network(net, inputs['test'], labels['test'],
                                              batch_size=128)
prt = ('test set:test_loss: {:0.4f}, test_acc: {:0.4f}')
print(prt.format(test_loss,test_acc))
