import numpy as np
from PIL import Image
from numpy import genfromtxt
import numpy
from glob import glob
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

numpy.set_printoptions(threshold=numpy.inf)

def pca_reconstruction(X):
    #binarizing the data
    X = X > 0
    X = 1*X
    for i in range(500):
    	for j in range(2601):
    		if(X[i][j] > 0 and X[i][j] < 1):
    			print("data not binarized")

    # print(X[0])
    #setting number of pca components to extract
    pca_comp = 50
    
    pca = PCA(n_components=pca_comp).fit(X)
    eigarr = pca.explained_variance_ratio_
    array = pca.components_
    # print "components size : " + str(array.shape)
    # print eigarr.shape
    # code to show cumulative variance vs number of components graph
    arr = np.arange(1,pca_comp + 1,1)
    eigarr = eigarr * 100
    eigarr = np.cumsum(eigarr)
    plt.plot(arr, eigarr)
    plt.xlabel("Component")
    plt.ylabel("Cumulative variance")
    plt.title("Components vs. cumulative variance")
    plt.show()

    x_pca = pca.transform(X)
    # print x_pca.shape
    transformed_x_pca = x_pca.T
    print(transformed_x_pca.shape)
    arr = np.arange(1,501,1)
    plt.plot(arr, transformed_x_pca[5])
    # plt.plot(arr, transformed_x_pca[1])
    plt.show()
    x_original = pca.inverse_transform(x_pca)
    print("x_original shape : " + str(x_original.shape))
    
    # code to show reconstructed images
    fig, axs = plt.subplots(2, 10)
    for i in range(10):
        axs[0][i].imshow(X[i, :].reshape(51, 51))
        axs[1][i].imshow(x_original[i, :].reshape(51, 51))
    # fig.show()
    # plt.show()

    return 

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def dir_to_dataset(glob_files):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=numericalSort) ):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        #print("\t %s files processed"%file_count)
	#print file_name
    return np.array(dataset)

cwd = os.getcwd()
print(str(cwd))
dataset = dir_to_dataset("*.png")
# print("dataset : " + str(dataset))
print("dataset shape " + str(dataset.shape))
pca_reconstruction(dataset)
