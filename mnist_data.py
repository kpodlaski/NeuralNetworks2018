import os
import struct
import gzip
import numpy as np
import collections
import gzip



import scipy.misc as scipy_misc

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "files\\"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image
    """
    path= os.path.join(os.path.dirname(os.path.realpath(__file__)),path)
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError ("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    print(os.path.dirname(os.path.realpath(__file__)))
    print(fname_lbl)
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        print(magic)
        print(num)
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII",
		fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def ascii_show(image):
    _row =0
    for lbl, y in image:
         row = ""
         row2 = ""
         print('{0: <3}'.format(str(lbl)) + "_")
         for x in y:
            for el in x:
                if el > 10 : row2+="*"
                else : row2+=" "
            print(row2)
            row2=""
         #print(row)
         print(row2)
         _row=_row+1
         if _row > 28 : break

def read_img_from_file(fname):
    img = scipy_misc.imread(fname,True,'L')
    return img

#train-images-idx3-ubyte.gz

def download_mnist(images_filename, labels_filename):
    if ~os.path.isfile("in_files/"+images_filename):
        print("download "+images_filename)
        url = 'http://yann.lecun.com/exdb/mnist/'+images_filename+".gz"
        import requests
        r = requests.get(url, allow_redirects=True)
        with open("in_files/"+images_filename, 'wb') as f:
            f.write(gzip.decompress(r.content))
            f.close()
    if ~os.path.isfile("in_files/"+labels_filename):
        print("download " + labels_filename)
        url = 'http://yann.lecun.com/exdb/mnist/'+labels_filename+".gz"
        r = requests.get(url, allow_redirects=True)
        with open("in_files/"+labels_filename, 'wb') as f:
            f.write(gzip.decompress(r.content))
            f.close()

#download_mnist('train-images-idx3-ubyte','train-labels-idx1-ubyte')
#download_mnist('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')
mnist = read("training",path="in_files\\")
ascii_show(mnist)