import numpy as np
import matplotlib.pyplot as plt


# utility functions
def show(img):
    """ Display a sample image combining two images"""
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()


def construct_operations(n: int = 28):
    """ Constructs operation pattern tensors. '+' for addition 
        and '-'  for susbstraction. """

    mid = n // 2
    sum_op = np.zeros((n, n), dtype=np.float32)
    sum_op[mid-1:mid+1, :] = 1
    sum_op[:, mid-1:mid+1] = 1
    sum_op = np.expand_dims(sum_op, axis=(0, 1))

    diff_op = np.zeros((n, n), dtype=np.float32)
    diff_op[mid-1:mid+1, :] = 1
    diff_op = np.expand_dims(diff_op, axis=(0, 1)) 

    return [sum_op, diff_op]
