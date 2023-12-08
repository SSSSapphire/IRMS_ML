from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained

#X = np.array([[1, 2], [1, 4],[1, 0], [4, 2], [4, 4], [4, 0]])
n_cluster = 2

def doKmeansConstrained(X,firstList):
   cluster = KMeansConstrained(
      n_clusters=n_cluster,
      size_min=2,
      size_max=20,
      random_state=0
   )
   pre = cluster.fit_predict(X)

   centroid = cluster.cluster_centers_
   centroid
   centroid.shape

   inertia = cluster.inertia_
   inertia

   color = ["green","blue"]
   fig, ax2 = plt.subplots(1)
   tempCounter = 0
   for i in range(n_cluster):
      print("zzzzzzzzzzzzzzzzzzz", n_cluster,i,pre)
      ax2.scatter(X[pre==i, 0], X[pre==i, 1]
           ,marker='o'
           ,s=8
           ,c=color[i]
           )
      if(i==0):
            for label,x,y in zip(firstList,X[pre==0, 0],X[pre==0, 1]):
                plt.text(x,y,label)
                tempCounter = tempCounter + 1
      if(i==1):
            firstList1 = firstList[tempCounter:firstList.size] 
            for label,x,y in zip(firstList1,X[pre==1, 0],X[pre==1, 1]):
                plt.text(x,y,label)
      ax2.scatter(centroid[:,0], centroid[:,1]
         ,marker="x"
         ,s=15
         ,c="black"
         )
   plt.show()

#clf = KMeansConstrained(
 #   n_clusters=n_cluster,
  #  size_min=2,
  #  size_max=5,
  #  random_state=0
#)
#pre = clf.fit_predict(X)
#clf.cluster_centers_
#centroid = clf.cluster_centers_
#centroid
#centroid.shape

#color = ["red","blue"]
#fig, ax1 = plt.subplots(1)

#for i in range(n_cluster):
#        ax1.scatter(X[pre==i, 0], X[pre==i, 1]
#          ,marker='o'
#          ,c=color[i]
#           )
#        ax1.scatter(centroid[:,0],centroid[:,1]
 #          ,marker="x"
 #          ,s=15
 #          ,c="black")
#plt.show()