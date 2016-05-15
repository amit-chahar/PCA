from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.inf)


import math

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


no_of_images = 500
no_of_dims = 2601
imageArr = np.zeros(shape=(no_of_dims, no_of_images))
# print("shape of imageArr : " + str(imageArr.shape))
dataset = []
# A = []
for i in range(1, 501):
	filename = str(i) + ".png"
	img = Image.open(filename).convert('L')
	# print("shpae of img : " + str(list(img.getdata())))
	# pixels = [f[0] for f in list(img.getdata())]
	# A = np.array(img)
	# print("size of A : " + str(A.shape))
	imageArr[:,i-1:i] = np.array(img).reshape(no_of_dims, 1)

# print("shape of imageArr : " + str(imageArr[:,0]))
# imageArr = 1 * (imageArr[:,:] > 127)
# A = A.astype(int)
# print(str(A))
# plt.imshow(A.reshape(51, 51), cmap='Greys_r')
# plt.show()
# mean_vector = np.mean(imageArr, axis=1)
# print("mean_vector shape : " + str(mean_vector))
# assert mean_vector.shape == (no_of_dims,)

cov_mat = np.cov(imageArr)
# print('Scatter Matrix:\n', cov_mat[:,1000])
eigval, eigvec = np.linalg.eig(cov_mat)
sortedind = np.argsort(eigval)
sortedind = sortedind[::-1]
# print("eigval " + str(eigval[sortedind[0:20]]))


n_comp = 6
n_comp_mat = np.zeros(shape=(2601, n_comp))
# n_comp_mat[:,n_comp - 1:n_comp] = eigvec[:,sortedind[n_comp - 1:n_comp]]
n_comp_mat[:,0:n_comp] = eigvec[:,sortedind[0:n_comp]]
# print("n_comp_mat size : " + str(n_comp_mat.shape))
output = n_comp_mat.T.dot(imageArr[:, :])
# print("output size : " + str(output.shape))
print(output.shape)

file = open("theta.txt",'r')
abc = list(file.read().split("\n"))
abc = [float(i) for i in abc]
from scipy import stats
pc1 = output[0]
pc2 = output[1]
pc3 = output[2]
pc4 = output[3]
pc5 = output[4]
pc6 = output[5]
data = np.array([[e for e in pc1], [e for e in pc2]])
print(np.corrcoef(data))
print(len(abc))
print("(pc1,angles)" + str(pearson_def(pc1,abc)))
print("(pc2,angles)" + str(pearson_def(pc2,abc)))
print("(pc3,angles)" + str(pearson_def(pc3,abc)))
print("(pc4,angles)" + str(pearson_def(pc4,abc)))
print("(pc5,angles)" + str(pearson_def(pc5,abc)))
print("(pc6,angles)" + str(pearson_def(pc6,abc)))
# print("(pc2,pc3)" + str(pearson_def(pc2,pc3)))
# print("(pc2,pc4)" + str(stats.pearsonr(pc2,pc4)))
# print("(pc2,pc5)" + str(stats.pearsonr(pc2,pc5)))
# print("(pc2,pc6)" + str(stats.pearsonr(pc2,pc6)))
# print("(pc3,pc4)" + str(stats.pearsonr(pc3,pc4)))
# print("(pc3,pc5)" + str(stats.pearsonr(pc3,pc5)))
# print("(pc3,pc6)" + str(stats.pearsonr(pc3,pc6)))
# print("(pc4,pc5)" + str(stats.pearsonr(pc4,pc5)))
# print("(pc4,pc6)" + str(stats.pearsonr(pc4,pc6)))
# print("(pc5,pc6)" + str(stats.pearsonr(pc5,pc6)))

print("cross correlation")
print("(pc1,angles)" + str(np.correlate(pc1,abc, "full")))
print("(pc2,angles)" + str(np.correlate(pc2,abc)))
print("(pc3,angles)" + str(np.correlate(pc3,abc)))
print("(pc4,angles)" + str(np.correlate(pc4,abc)))
print("(pc5,angles)" + str(np.correlate(pc5,abc)))
print("(pc6,angles)" + str(np.correlate(pc6,abc)))

# plt.imshow(output.reshape(3, 4))
# print("output : " + str(output))

a = imageArr[:,0:14]
Ar = np.dot(n_comp_mat,output)
# print(Ar.shape)
Ar = Ar.real
Ar.astype(int)
Ar = Ar.T
fig, axs = plt.subplots(2, 7)
for i in range(7):
        axs[0][i].imshow(Ar[i,:].reshape(51, 51),cmap='Greys_r' )
        axs[1][i].imshow(Ar[i+7, :].reshape(51, 51),cmap='Greys_r')
fig.show()
# plt.imshow(Ar.reshape(51,51),cmap='Greys_r')
plt.show()
