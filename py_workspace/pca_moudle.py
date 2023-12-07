import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix

tempArray = np.array([['C_D1_0_2'],['C_D1_1'],['C_D1_2'],['C_D1_3'],['C_D1_4']])
n_components = 31
random_state = 9527
pca = PCA(n_components=n_components, 
          random_state=random_state)
def do_Pca(Z):
    pca_2d = PCA(2, random_state=random_state)
    L = pca_2d.fit_transform(Z)
    return L

def do_Kmeans(Z,firstList):
    print(firstList.index)
    n_cluster = 2
    random_state = 0
    cluster =  KMeans(n_clusters = n_cluster, n_init= "auto",random_state = random_state).fit(Z)
    #查看每个样本对应的类
    y_pred = cluster.labels_
    y_pred
    #使用部分数据预测质心
    pre = cluster.fit_predict(Z)
    pre == y_pred
    #质心
    centroid = cluster.cluster_centers_
    centroid
    centroid.shape
    #总距离平方和
    inertia = cluster.inertia_
    inertia
    print("轮廓系数silhouette_score(0-1，越高越好) = " + str(silhouette_score(Z,y_pred)))
    print("卡林斯基哈拉巴斯指数calinski_harabasz_score(越高越好) = " + str(calinski_harabasz_score(Z,y_pred)))
    print("戴维斯布尔丁指数davies_bouldin_score(越小越好) = " + str(davies_bouldin_score(Z,y_pred)))
    #print("权变矩阵contingency_matrix如下")
    #print(contingency_matrix(Z,y_pred)
    
    color = ["red","blue"]
    fig, ax1 = plt.subplots(1)
    tempCounter = 0
    for i in range(n_cluster):
        ax1.scatter(Z[y_pred==i, 0], Z[y_pred==i, 1]
           ,marker='o'
           ,s=8
           ,c=color[i]
           )
        print("i=",i)
        if(i==0):
            for label,x,y in zip(firstList,Z[y_pred==0, 0],Z[y_pred==0, 1]):
                plt.text(x,y,label)
                tempCounter = tempCounter + 1
        if(i==1):
            firstList1 = firstList[tempCounter:firstList.size] 
            for label,x,y in zip(firstList1,Z[y_pred==1, 0],Z[y_pred==1, 1]):
                plt.text(x,y,label)
    ax1.scatter(centroid[:,0],centroid[:,1]
        ,marker="x"
        ,s=15
        ,c="black")
    plt.show()

    #因实验本身的目的，通过改变n_cluster分簇数量来影响inertia来评估分类数量效果不好
    #轮廓系数silhouette_score对聚类的评估有一定参考性
    #卡林斯基哈拉巴斯指数calinski_harabasz_score，优点快
    #戴维斯布尔丁指数davies_bouldin_score，优点：评价准确度相对高，稳定性好；缺点：较慢，对质点的选择敏感
    #对新加入的数据进行预测，以已有的数据所构建的模型
    #kmeans.predict([[0, 0], [1, 1]])
    