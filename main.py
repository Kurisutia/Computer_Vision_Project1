import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.decomposition import PCA

from layers import Linear, relu, SoftmaxCrossEntropyLoss,tanh,sigmoid,leakyrelu
from network import Network

def main():
    
    inputs, labels =  load_data()
    epoch=12
    np.random.seed()
    lr=np.random.randint(10,50,500)
    lr=lr/100
    ld=np.random.randint(1,100,500)
    ld=ld/1000
    hd=np.random.randint(10,40,500)
    hd=hd*20
    max_acc=0
    flag=-1
    pt = ('search times:{:3} \nparameters:learing_rate: {:0.3f}, lamda: {:0.4f},hide_layer:{:3}')
    for i in range(100):
        print(pt.format(i,lr[i], ld[i],hd[i]))
        train_loss_set,val_loss_set,val_acc_set,net=Train_Net(inputs,labels,lr[i],ld[i],hd[i],epoch)
        acc=val_acc_set[-1]
        if acc>max_acc:
            max_acc=acc
            flag=i;
            train_loss=train_loss_set
            val_loss=val_loss_set
            val_acc=val_acc_set
            #save the model
            with open("net",'wb') as f:
                pickle.dump(net, f)
    prnt = ('found parameters:learing_rate: {:0.3f}, lamda: {:0.4f},hide_layer:{:3},max acc:{:0.4f}')
    print(prnt.format(lr[flag], ld[flag],hd[flag],max_acc))
    
    num=int(epoch*6)
    x_=np.arange(0,num,1)
    plt.plot(x_,train_loss,'g-',label='train loss')
    plt.plot(x_,val_loss,'b-',label='val loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.plot(x_,val_acc,'r-',label='val acc')
    plt.legend(loc='upper left')
    plt.show()
    
    #read model
    with open("net",'rb') as f:
        net=pickle.load(f)
    #plot
    w=[]
    for layer in net.layers:
        if layer.func=="linear":
            w.append(layer.W)
    model_pca = PCA(n_components=3)
    for i in [0,1]:
        w[i]= model_pca.fit(w[i]).transform(w[i])
    w[0]=abs(w[0])
    w[1]=abs(w[1])
    wid=int(hd[flag]/20)
    w[0]=w[0].reshape(28,28,3)
    w[1]=w[1].reshape(20,wid,3)
    plt.imshow(w[0])
    plt.show()
    plt.imshow(w[1])
    plt.show() 
    
    #test
    test(net,inputs['test'],labels['test'])
    
def test(net,x_test,y_test):    
    #test
    test_loss, test_acc = validate_network(net, x_test, y_test,
                                              batch_size=128)
    prt = ('test set:test_loss: {:0.4f}, test_acc: {:0.4f}')
    print(prt.format(test_loss,test_acc))
    
def Train_Net(inputs,labels,learning_rate,lamda,hide,epoch):
    # np.random.seed()
    n_classes = 10
    dim = 784
    batch_size=128
    net = Network(learning_rate,lamda)
    net.add_layer(Linear(dim, hide))
    net.add_layer(relu())
    net.add_layer(Linear(hide,n_classes))
    net.set_loss(SoftmaxCrossEntropyLoss())

    train_loss_set,val_loss_set,val_acc_set=train_network(net, inputs, labels,epoch,batch_size)
    print('=============================================')
    return train_loss_set,val_loss_set,val_acc_set,net
    

def load_data(): 
    (x,y),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
    tr=x.shape[0]
    te=x_test.shape[0]
    x=x.reshape(tr,784)
    x_test=x_test.reshape(te,784)
    
    x_mean=np.mean(x)
    x_std=np.std(x)
    x=(x-x_mean)/x_std
    x_test_mean=np.mean(x_test)
    x_test_std=np.std(x_test)
    x_test=(x_test-x_test_mean)/x_test_std
    
    train_data=dict()
    train_label=dict()
    
    sq=int(tr*0.7)
    #print(sq)
    
    x_train=x[:sq]
    y_train=y[:sq]
    
    x_val=x[sq:]
    y_val=y[sq:]
    
    train_data["train"]=x_train
    train_data["val"]=x_val
    train_data["test"]=x_test
    
    train_label['train']=y_train
    train_label['val']=y_val
    train_label['test']=y_test
    
    return train_data,train_label


def validate_network(network, inputs, labels, batch_size):
    '''
    Calculates loss and accuracy for network when predicting labels from inputs

    Args:
        network (Network): A neural network
        inputs (numpy.ndarray): Inputs to the network
        labels (numpy.ndarray): Labels corresponding to inputs
        batch_size (int): Minibatch size

    Returns:
        avg_loss, accuracy
        avg_loss (float): The average loss per sample using the loss function
                          specified in network
        accuracy (float): (Correct predictions) / (number of samples)
    '''
    n_inputs = inputs.shape[0]

    tot_loss = 0.0
    tot_correct = 0
    start_idx = 0
    while start_idx < n_inputs:
        end_idx = min(start_idx+batch_size, n_inputs)
        mb_inputs = inputs[start_idx:end_idx]
        mb_labels = labels[start_idx:end_idx]

        scores = network.predict(mb_inputs)
        W=network.layers[-1].W
        loss, _ = network.loss.get_loss(scores, mb_labels,network)
        tot_loss += loss * (end_idx-start_idx)
        preds = np.argmax(scores, axis=1)
        tot_correct += np.sum(preds==mb_labels)

        start_idx += batch_size

    avg_loss = tot_loss / n_inputs
    accuracy = tot_correct / n_inputs

    return avg_loss, accuracy


def train_network(network, inputs, labels, n_epochs, batch_size=128):
    '''
    Trains a network for n_epochs

    Args:
        network (Network): The neural network to be trained
        inputs (dict): Dictionary with keys 'train' and 'val' mapping to numpy
                       arrays
        labels (dict): Dictionary with keys 'train' and 'val' mapping to numpy
                       arrays
        n_epochs (int): Specifies number of epochs trained for
        batch_size (int): Number of samples in a minibatch
    '''
    train_inputs = inputs['train']
    train_labels = labels['train']

    n_train = train_inputs.shape[0]
    train_loss_set=[]
    val_loss_set=[]
    val_acc_set=[]
    
    part=np.linspace(0,n_train,7)
    part=part.astype(int)
    train_p=[]
    label_p=[]

    # Train network
    for epoch in range(n_epochs):
        rd = np.random.permutation(n_train)
        train_inputs=train_inputs[rd]
        train_labels=train_labels[rd]
        for i in range(6):
            train_p.append(train_inputs[part[i]:part[i+1]])   
            label_p.append(train_labels[part[i]:part[i+1]])
        for i in range(6):
            p_train=train_p[i].shape[0]
            order = np.random.permutation(p_train)
            num_batches = p_train // batch_size
            train_loss = 0
            start_idx = 0
            while start_idx < p_train:
                end_idx = min(start_idx+batch_size, p_train)
                idxs = order[start_idx:end_idx]
                mb_inputs = train_p[i][idxs]
                mb_labels = label_p[i][idxs]
                #train minibatch
                train_loss += network.train(mb_inputs, mb_labels,epoch)
                start_idx += batch_size 
            avg_train_loss = train_loss/num_batches
            avg_val_loss, val_acc = validate_network(network, inputs['val'],
                                                             labels['val'], batch_size)
            train_loss_set.append(avg_train_loss)
            val_loss_set.append(avg_val_loss)
            val_acc_set.append(val_acc)

            prnt_tmplt = ('Epoch: {:3},part:{:2} train loss: {:0.4f},val loss:{:0.4f},val acc:{:0.4f}')
            print(prnt_tmplt.format(epoch, i,avg_train_loss,avg_val_loss,val_acc))
            

    return train_loss_set,val_loss_set,val_acc_set

if __name__ == '__main__':
    main()
