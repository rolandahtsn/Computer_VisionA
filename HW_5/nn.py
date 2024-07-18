import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    b = np.sqrt(6/(in_size+out_size))
    W = np.random.uniform(-b,b,(in_size,out_size))

    b = np.zeros((out_size))
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function

def sigmoid(x):
    res = None
    res = 1/(1+np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None

    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    #Activation..
    pre_act = np.dot(X,W)
    pre_act = pre_act+b

    #Every value in pre_act
    post_act = activation(pre_act)


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    #get max x [examples,classes]
    x_max = np.max(x,axis=1,keepdims=True) #Keepdims will return a 2D array, when not using this parameter cannot perform subtraction of matrix
    shifted_x = x-x_max
    x_exp = np.exp(shifted_x)

    x_sum = np.exp(shifted_x).sum(axis=1, keepdims=True)
    res = x_exp/x_sum

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]

def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    acc = []
    loss = 0
    

    for i in range (0,y.shape[0]):
       #Compute loss
       loss = loss-np.sum(y[i]*np.log(probs[i]))
       probability = np.argmax(probs[i])
       y_2= np.argmax(y[i])

       #Find where probs and y are equal
       if probability == y_2:
            acc.append(True)
       else:
        acc.append(False)
    acc = np.mean(acc)

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # det = np.matmul(delta,activation_deriv(post_act))
    det = delta*activation_deriv(post_act)
    grad_W = np.dot(np.transpose(X),det)
    grad_b = np.sum(det, axis=0)
    grad_X = np.dot(det,np.transpose(W)) 

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    #Takes x and y as inputs and splits it into random batches
    batches_num = x.shape[0]/batch_size

    #Get random batches
    random_batches_x_y = np.random.choice(x.shape[0],y.shape[0],False)

    i = 0
    while i < batches_num:
        if len(random_batches_x_y) > batch_size:
            indexes = random_batches_x_y[0:batch_size]
            random_batches_x_y = random_batches_x_y[batch_size:]
        else:
            indexes = random_batches_x_y

        batches.append((x[indexes],y[indexes]))
        i+=1
        
    return batches
