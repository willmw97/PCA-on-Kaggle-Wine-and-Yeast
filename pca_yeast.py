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
from numpy import genfromtxt

#yeast = pd.read_csv("yeast.csv")
yeast = np.loadtxt(r'yeast.data',delimiter=',')
#Myyeast = genfromtxt('yeast.csv', delimiter=',')
#yeast = np.delete(Myyeast, 9, axis=1)
print(yeast)
yeast[:,:8] = yeast[:,:8]-yeast[:,:8].mean(axis=0)
imax = np.concatenate((yeast.max(axis=0)*np.ones((1,9)),yeast.min(axis=0)*np.ones((1,9))),axis=0).max(axis=0)
yeast[:,:8] = yeast[:,:8]/imax[:8]
labels = yeast[:,8:]
yeast = yeast[:,:8]

#label_names = ['CYT', 'ERL', 'EXC', 'ME1', 'ME2', 'ME3', 'MIT', 'NUC', 'POX', 'VAC']
#                  1,   2,      3,      4,  5,      6,      7,      8,   9,     10

order = list(range(np.shape(yeast)[0]))
np.random.shuffle(order)
yeast = yeast[order,:]
labels = labels[order,0]

w0 = np.where(labels==1)
w1 = np.where(labels==2)
w2 = np.where(labels==3)
w3 = np.where(labels==4)
w4 = np.where(labels==5)
w5 = np.where(labels==6)
w6 = np.where(labels==7)
w7 = np.where(labels==8)
w8 = np.where(labels==9)
w9 = np.where(labels==10)

pl.figure(1)
pl.title('Original Yeast Data')
pl.plot(yeast[w0,0],yeast[w0,1],'ok', color="green")
pl.plot(yeast[w1,0],yeast[w1,1],'ok',color="yellow")
pl.plot(yeast[w2,0],yeast[w2,1],'ok',color="blue")
pl.plot(yeast[w3,0],yeast[w3,1],'ok',color="navy")
pl.plot(yeast[w4,0],yeast[w4,1],'ok',color="lightgreen")
pl.plot(yeast[w5,0],yeast[w5,1],'ok',color="blueviolet")
pl.plot(yeast[w6,0],yeast[w6,1],'ok',color="red")
pl.plot(yeast[w7,0],yeast[w7,1],'ok',color="crimson")
pl.plot(yeast[w8,0],yeast[w8,1],'ok',color="indianred")
pl.plot(yeast[w9,0],yeast[w9,1],'ok',color="purple")

pl.figure(2)
x,y,evals,evecs = pca.pca(yeast,2)
print(evecs)
print(evals)

pl.title('After PCA')
pl.plot(y[w0,0],y[w0,2],'ok',color="green")
pl.plot(y[w1,0],y[w1,2],'ok',color="yellow")
pl.plot(y[w2,0],y[w2,2],'ok',color="blue")
pl.plot(y[w3,0],y[w3,2],'ok',color="navy")
pl.plot(y[w4,0],y[w4,2],'ok',color="lightgreen")
pl.plot(y[w5,0],y[w5,2],'ok',color="blueviolet")
pl.plot(y[w6,0],y[w6,2],'ok',color="red")
pl.plot(y[w7,0],y[w7,2],'ok',color="crimson")
pl.plot(y[w8,0],y[w8,2],'ok',color="indianred")
pl.plot(y[w9,0],y[w9,2],'ok',color="purple")
pl.show()



