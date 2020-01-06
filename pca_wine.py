
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Various dimensionality reductions running on the Iris dataset
import pylab as pl
import numpy as np
import pca


iris = np.loadtxt(r'wine.data',delimiter=',')
iris[:,1:] = iris[:,1:]-iris[:,1:].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,14)),iris.min(axis=0)*np.ones((1,14))),axis=0).max(axis=0)
iris[:,1:] = iris[:,1:]/imax[1:]
labels = iris[:,:1]
iris = iris[:,1:]

order = list(range(np.shape(iris)[0]))
np.random.shuffle(order)
iris = iris[order,:]
labels = labels[order,0]

w0 = np.where(labels==1)
w1 = np.where(labels==2)
w2 = np.where(labels==3)

pl.figure(1)
pl.title('Original Wine Data')
pl.plot(iris[:,0],iris[:,1],'ok')
pl.plot(iris[:,0],iris[:,1],'^k')
pl.plot(iris[:,0],iris[:,1],'vk')

pl.figure(2)
x,y,evals,evecs = pca.pca(iris,2)
print(evecs)
print(evals)


pl.title('After PCA')
pl.plot(y[w0,0],y[w0,2],'ok')
pl.plot(y[w1,0],y[w1,2],'^k')
pl.plot(y[w2,0],y[w2,2],'vk')
pl.show()

