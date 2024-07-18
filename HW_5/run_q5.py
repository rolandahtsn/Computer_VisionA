import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('./data/nist36_train.mat')
valid_data = scipy.io.loadmat('./data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1') #ReLU
initialize_weights(hidden_size, hidden_size, params, 'hidden1') #ReLU
initialize_weights(hidden_size, hidden_size, params, 'hidden2') #ReLU
initialize_weights(hidden_size, 1024, params, 'output') #Sigmoid

# Q5.1.2
keys = [key for key in params.keys()]
for key in keys:
    params['m_' + key] = np.zeros(params[key].shape)

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:

        #forward
        h = forward(xb,params,'layer1',relu)
        h_1 = forward(h,params,'hidden1',relu)
        h_2 = forward(h_1,params,'hidden2',relu)
        output_ = forward(h_2,params,'output',sigmoid)

        # loss
        p_X = np.square(output_-xb)
        loss = p_X.sum()

        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss

        # backward
        delta = output_ - xb
        delta_2 = backwards(2*delta, params, 'output', sigmoid_deriv)
        delta_3 = backwards(delta_2, params, 'hidden2', relu_deriv)
        delta_4 = backwards(delta_3, params, 'hidden1', relu_deriv)
        backwards(delta_4,params,'layer1',relu_deriv)
        
        #print(params.keys())
        params_w_list = ['Wlayer1','Wlayer2','Wlayer3','Woutput','Whidden1','Whidden2']
        params_b_list = ['blayer1','blayer2','blayer3','boutput','bhidden1','bhidden2']

        # apply gradient  (5.1.1)
        # #W Layer
        # for i in params_w_list:
        #     params[i] -=learning_rate*params['grad_'+ i]
        # #B layer
        # for j in params_b_list:
        #     params[j] -=learning_rate*params['grad_'+ j]

        #Momentum
        #just use 'm_'+name variables
        #W layer

        for i in params_w_list:
            params['m_'+i] = 0.9*params['m_'+i] - learning_rate*params['grad_'+i]
            params[i] += params['m_'+ i]
        #B layer
        for j in params_b_list:
            params['m_'+j] = 0.9*params['m_'+j] - learning_rate*params['grad_'+j]
            params[j] += params['m_'+ j]

    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
# both will be plotted below

#Code here
chosen_index = np.random.choice(10,size = 10,replace = False)
h = forward(visualize_x,params,'layer1',relu)
h_1 = forward(h,params,'hidden1',relu)
h_2 = forward(h_1,params,'hidden2',relu)
output = forward(h_2,params,'output',sigmoid)

reconstructed_x = []
for i in chosen_index:
    reconstructed_x.append(output[i])

# plot visualize_x and reconstructed_x
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio as psnr
# evaluate PSNR
high_PSNR = 0
for i in range(0,visualize_x.shape[0]):
    reshape_x = visualize_x[i].reshape((32,32))
    reshape_output = reconstructed_x[i].reshape((32,32))
    high_PSNR += psnr(reshape_x,reshape_output)
high_PSNR = high_PSNR/visualize_x.shape[0]
print(high_PSNR)