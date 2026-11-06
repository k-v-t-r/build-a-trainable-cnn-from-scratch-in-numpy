"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    index =0
    ans = []
    for row in matrix:
        maximum = row[-1]
        index = len(row) -1
        pointer = len(row) -1
        while pointer>=0:
            num=row[pointer]
            if maximum <= num:
                index = pointer
                maximum = num
            pointer = pointer -1
        ans.append(index)
    return np.array(ans)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    ans = []
    for row in matrix:
        temp = []
        temp.append(max(row))
        ans.append(temp)
    return np.array(ans)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    ans =[]
    for row in matrix:
        temp = []
        temp.append(sum(row))
        ans.append(temp)
    return np.array(ans)

# Step 4 - exp_shifted
import numpy as np

def row_max(matrix):
    ans = []
    for row in matrix:
        temp = []
        temp.append(max(row))
        ans.append(temp)
    return np.array(ans)

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    # print(type(logits))
    # print(len(logits))
    # print(len(logits[0]))
    
    substracted = []
    maxnum = row_max(logits)
    i =0
    for row in logits:
        maxnumber = maxnum[i][0]
        i = i+1
        temp = []
        for num in row:
            temp.append(num - maxnumber)
        substracted.append(np.exp(temp))
    
    return np.array(substracted)

# Step 5 - stable_softmax
def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    ans =[]
    for row in matrix:
        temp = []
        temp.append(sum(row))
        ans.append(temp)
    return np.array(ans)

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    # print(type(logits))
    # print(len(logits))
    # print(len(logits[0]))
    
    substracted = []
    maxnum = row_max(logits)
    i =0
    for row in logits:
        maxnumber = maxnum[i][0]
        i = i+1
        temp = []
        for num in row:
            temp.append(num - maxnumber)
        substracted.append(np.exp(temp))
    
    return np.array(substracted)

def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    logits = exp_shifted(logits)
    sumarray = row_sum(logits)
    # print(sumarray)
    for i in range(0,len(logits)):
        sum_of_row = sumarray[i][0]
        for j in range(0,len(logits[0])):
            logits[i][j] = logits[i][j] / sum_of_row
    return logits

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    # print(labels)
    # print(num_classes)
    ans =np.zeros((len(labels),int(num_classes)),dtype=float)
    for i in range(0,len(labels)):
        ans[i][labels[i]]=1

    return ans

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    arr = []
    pointer = 0
    for i in labels:
        arr.append(probs[pointer][i])
        pointer = pointer +1
    return np.array(arr)

# Step 8 - cross_entropy_loss
import numpy as np

def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    arr = []
    pointer = 0
    for i in labels:
        arr.append(probs[pointer][i])
        pointer = pointer +1
    return np.array(arr)

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    output = gather_true_class_probs(probs,labels)
    logsum =0
    for prob in output:
        if prob ==0:
            prob=eps
        logsum = logsum +np.log(prob)
    return (-1)*(logsum/len(labels))

# Step 9 - accuracy
def argmax_rows(matrix):
    index =0
    ans = []
    for row in matrix:
        maximum = row[-1]
        index = len(row) -1
        pointer = len(row) -1
        while pointer>=0:
            num=row[pointer]
            if maximum <= num:
                index = pointer
                maximum = num
            pointer = pointer -1
        ans.append(index)
    return np.array(ans)
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    output = argmax_rows(logits_or_probs)
    cnt = 0
    for i in range(0,len(output)):
        if output[i]==labels[i]:
            cnt = cnt+1
    return cnt/len(labels)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    std = np.sqrt(2 / fan_in)
    arr = np.random.randn(*shape) * std
    return arr

# Step 12 - init_zero_bias (not yet solved)
# TODO: implement

# Step 13 - pad_2d (not yet solved)
# TODO: implement

# Step 14 - output_spatial_size (not yet solved)
# TODO: implement

# Step 15 - im2col (not yet solved)
# TODO: implement

# Step 16 - col2im (not yet solved)
# TODO: implement

# Step 17 - conv2d_forward (not yet solved)
# TODO: implement

# Step 18 - conv2d_grad_input (not yet solved)
# TODO: implement

# Step 19 - conv2d_grad_weights (not yet solved)
# TODO: implement

# Step 20 - conv2d_grad_bias (not yet solved)
# TODO: implement

# Step 21 - conv2d_backward (not yet solved)
# TODO: implement

# Step 22 - maxpool2d_forward (not yet solved)
# TODO: implement

# Step 23 - scatter_grad_window (not yet solved)
# TODO: implement

# Step 24 - maxpool2d_backward (not yet solved)
# TODO: implement

# Step 25 - relu_forward (not yet solved)
# TODO: implement

# Step 26 - relu_backward (not yet solved)
# TODO: implement

# Step 27 - flatten_forward (not yet solved)
# TODO: implement

# Step 28 - flatten_backward (not yet solved)
# TODO: implement

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

