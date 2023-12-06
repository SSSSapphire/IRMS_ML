import csv
import array as arr
import pandas as pd
import pca_moudle
import numpy as np
import kmeans_constrained_moudle
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#excel转csv
xl_data = pd.read_excel('1113_31S_Con.xlsx')
xl_data.to_csv('1113_31S_Con.csv', index=False)

#读取csv文件存为df
df = pd.read_csv('1113_31S_Con.csv',index_col="Identifier")
np.savetxt("transCSV.csv",df,'%.18e',delimiter=' ')
print("原始数据:")
display(df)

#插值填补
print("中位数插值填补")
imp_mid = SimpleImputer(missing_values=np.NaN,strategy='median')
imputerDf = imp_mid.fit_transform(df)
display(imputerDf)
print("df.shape", df.shape)
np.savetxt("ImputerDf.csv",imputerDf,'%.18e',delimiter=' ')

#提取Identifier序列
print("提取序列")
firstList = xl_data['Identifier']
print(firstList)
print("次序列")
for i in range(30):
    print(firstList[1,i])


#print("平均差与标准差")
df_stats = df.describe().loc[['mean','std']]
df_stats.style.format("{:.2f}")
#display(df_stats)

#归一化
scaler = StandardScaler()
Z_sk = scaler.fit_transform(imputerDf)
print("归一化结果")
display(Z_sk)
np.savetxt("Normalazation.csv",Z_sk,'%.18e',delimiter=' ')

print("PCA降维结果")
pcaResult = pca_moudle.do_Pca(Z_sk)
display(pcaResult)
plt.scatter(pcaResult[:, 0], pcaResult[:, 1])
plt.axis('equal')
for i in range(31):
     plt.annotate(firstList[:, 1] ,(pcaResult[:, 0], pcaResult[:, 1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()

print("Kmeans聚类")
Kmeans = pca_moudle.do_Kmeans(pcaResult,firstList)

print("KmeansConstrained聚类")
#Kmeans_constrained = kmeans_constrained_moudle.doKmeansConstrained(L)