import array as arr
from operator import index
import pandas as pd
import pca_moudle
import tSNE_moudle
import k_means_moudle
import kmeans_constrained_moudle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer    


#excel转csv
xl_data = pd.read_excel('1113_31S_Con.xlsx')
xl_data.to_csv('1113_31S_Con.csv', index=False)

#读取csv文件存为df
df = pd.read_csv('1113_31S_Con.csv',index_col="Identifier")
np.savetxt("Temp/transCSV.csv",df,'%.18e',delimiter=' ')
print("原始数据:")
display(df)

#插值填补
print("中位数插值填补")
imp_mid = SimpleImputer(missing_values=np.NaN,strategy='median')
df_Impute = imp_mid.fit_transform(df)
display(df_Impute)
print("df.shape", df.shape)
np.savetxt("Temp/df_Impute.csv",df_Impute,'%.18e',delimiter=' ')

#提取Identifier序列
print("提取序列")
firstList = xl_data['Identifier']
print(firstList)
print(type(firstList))
print("次序列")
for i in range(firstList.size):
    print(firstList[i])

#print("平均差与标准差")
#df_stats = df.describe().loc[['mean','std']]
#df_stats.style.format("{:.2f}")

#归一化
scaler = StandardScaler()
df_Normal = scaler.fit_transform(df_Impute)
print("归一化结果")
display(df_Normal)
np.savetxt("Temp/Normalization.csv",df_Normal,'%.18e',delimiter=' ')

#PCA降维
print("PCA降维结果")
<<<<<<< HEAD
pcaResult = pca_moudle.do_Pca(Z_sk)
display(pcaResult)
print(type(pcaResult))
plt.subplot(1,2,1)
plt.scatter(pcaResult[:, 0], pcaResult[:, 1])
plt.axis('equal')
plt.title('PCA_Result')
for label,x,y in zip(firstList,pcaResult[:, 0],pcaResult[:, 1]):
=======
pca_Result = pca_moudle.do_Pca(df_Normal)
display(pca_Result)
print(type(pca_Result))
plt.scatter(pca_Result[:, 0], pca_Result[:, 1])
plt.axis('equal')
for label,x,y in zip(firstList,pca_Result[:, 0],pca_Result[:, 1]):
>>>>>>> 612777b601a0b9b511b925b786d354c96477a204
    plt.text(x,y,label)

#组装pcaResult与firstList为DataFrame
df_firstList = pd.DataFrame({'pointName':firstList.values})
df_PcaLabelLocation = pd.DataFrame(pca_Result)
df_PcaLabelLocation.columns = ['scatter_X','scatter_Y']
df_PcaLabelLocation['Scatter_Index'] = df_firstList['pointName']
print(df_PcaLabelLocation)

#tSNE降维
print("tSNE降维结果")
<<<<<<< HEAD
tSNE_Result = tSNE_moudle.do_tSNE(Z_sk)
plt.subplot(1,2,2)
plt.scatter(tSNE_Result[:, 0],tSNE_Result[:, 1])
plt.axis('equal')
plt.title('tSNE_Result')
plt.show()
=======
tSNE_Result = tSNE_moudle.do_tSNE(df_Normal)
>>>>>>> 612777b601a0b9b511b925b786d354c96477a204


#合并为series
#transFirstListDF = firstList.to_frame()
#transFirstListDF['index2'] = pca_Result[0]
#transFirstListDF = transFirstListDF.set_index('index2')

#display(transFirstListDF)


print("Kmeans聚类")
<<<<<<< HEAD
Kmeans_PCA = k_means_moudle.do_Kmeans(pcaResult,firstList)
Kmeans_tSNE = k_means_moudle.do_Kmeans(tSNE_Result,firstList)
=======
Kmeans_PCA = k_means_moudle.do_Kmeans(pca_Result,firstList,df_PcaLabelLocation)
#Kmeans_tSNE = k_means_moudle.do_Kmeans(tSNE_Result,firstList)
>>>>>>> 612777b601a0b9b511b925b786d354c96477a204


print("KmeansConstrained聚类")
#Kmeans_constrained = kmeans_constrained_moudle.doKmeansConstrained(pca_Result,firstList)