import pandas as pd
import pca_moudle
import numpy as np
import kmeans_constrained_moudle
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('X1017_C+Z2.csv',index_col="Identifier")
print("df.shape", df.shape)

print("原始数据:")
display(df)

#print("平均差与标准差")
df_stats = df.describe().loc[['mean','std']]
df_stats.style.format("{:.2f}")
#display(df_stats)

#归一化
#X = df.iloc[:,1:]
scaler = StandardScaler()
Z_sk = scaler.fit_transform(df)
print("归一化结果")
display(Z_sk)
np.savetxt("Normalazation.csv",Z_sk,'%.18e',delimiter=' ')

print("PCA降维结果")
L = pca_moudle.do_Pca(Z_sk)
display(L)
plt.scatter(L[:, 0], L[:, 1])
plt.axis('equal')
plt.show()

print("Kmeans聚类")
Kmeans = pca_moudle.do_Kmeans(L)

print("KmeansConstrained聚类")
Kmeans_constrained = kmeans_constrained_moudle.doKmeansConstrained(L)