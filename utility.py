import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

def load_img():
    """Loads the MNIST train images normalised between 0 and 1."""
    df = pd.read_csv('../Data/train.csv')
    df = df.drop('label', axis = 1)
    df = df.as_matrix()/255
    return df

def display(image, image_width = 28, image_height = 28):
    """Reshape a 1D numpy array and display it using matplotlib"""
    disp = image.reshape([image_width, image_height])
    plt.axis('off')
    plt.imshow(disp, cmap = matplotlib.cm.binary)
    plt.show()