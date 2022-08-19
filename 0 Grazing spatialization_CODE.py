"""
@File  : 0 Grazing spatialization_CODE.py
@Author: Nan Meng
@email  : nanmeng_st@rcees.ac.cn
@Date  : 2022/6/13 18:09
"""

from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
import sklearn
import numpy as np
import jenkspy
import random
from pandas.core.frame import DataFrame
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import catboost as cb
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import gaussian_kde
from RasterRead import IMAGE
drv = IMAGE()
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy import stats
import glob

"========================================= [1] Get the best groups by using AD index ========================================="
# Dir_raster="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 IndependentVariable/"
# Dir_county="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 CountyValue/"
# Dir_SU="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 DependentVariable/"
# Dir_output="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Simple Spatialization/"
# for yeari in range(1982,2016,1):
#     year=int(yeari)
#     SU = pd.read_csv(Dir_SU+"SU1982-SU2015.csv") # SU is the sheep unit(SU) at the county-level in yeari(34 years)
#     SU = np.array(SU)[:,year-1982+1].reshape(-1,1)
#     proj, geos, COUNTY = drv.read_img(Dir_county+"COUNTY_VALUE.tif")  # COUNTY is the ID of county (244 counties)
#     proj, geos, D = drv.read_img(Dir_raster+"18 D_RF_"+str(yeari)+".tif")  # D is the AD index in yeari
#     COUNTY,D = np.array(COUNTY).reshape(-1,1),np.array(D).reshape(-1,1)
#     DATA=np.concatenate((COUNTY,D),axis=1)
#     SU_RASTER=[]  # The SU at pixel-level
#     Group_RASTER=[]  # The group we want to get in this step
#     for ID in range(1,245):
#         DATA1=[]
#         for i in range(0,DATA.shape[0]):
#             if DATA[i,0]==ID:
#                 DATA1.append(1)
#             else:
#                 DATA1.append(0)
#         DATA1=np.array(DATA1).reshape(-1,1)
#         DATA_ID=DATA1*np.array(DATA[:,1]).reshape(-1,1)
#         SU_ID=SU[ID-1,0]
#         SUM=np.sum(DATA_ID)
#         SUi=SU_ID*DATA_ID/(SUM+0.001)
#         print(yeari,"年: 县域ID为",ID,"， 干扰度总和为",SUM,"， 该县域总羊单位为",SU_ID)
#         SU_RASTER.append(SUi.flatten())
#         SUi_remove0=SUi[SUi!=0]  # When the number of grids with non-zero values in a county is greater than 5, clustering is performed
#         print("the number of non-zero values is ", SUi_remove0.shape[0])
#         count=SUi_remove0.shape[0]
#         if count>5:
#             score=[]
#             for clusteri in range(2,6):
#                 clustering = AgglomerativeClustering(linkage='ward', n_clusters=clusteri)
#                 res = clustering.fit(SUi_remove0.reshape(-1,1).data)
#                 labels=clustering.labels_
#                 labels_count=pd.Series(clustering.labels_).value_counts()
#                 DBI=sklearn.metrics.davies_bouldin_score(SUi_remove0.reshape(-1,1).data, labels)
#                 score.append(DBI)
#             score=np.array(score)
#             score_min = np.min(score)
#             best_group = [k+2 for k in range(len(score)) if score[k]==score_min]  # Get the number of best groups
#             print("the number of best groups：",best_group[0])
#             clustering = AgglomerativeClustering(linkage='ward', n_clusters=best_group[0])
#             res = clustering.fit(SUi_remove0.reshape(-1,1).data)
#             groups = ID*10+clustering.labels_
#             labels_count = pd.Series(clustering.labels_).value_counts()
#             DBI = sklearn.metrics.davies_bouldin_score(SUi_remove0.reshape(-1,1).data, groups)
#         else:
#             groups = ID*10+np.zeros(SUi_remove0.shape[0])
#         A=SUi
#         MASK= A!=0
#         print(A.shape,MASK.shape)
#         A[MASK] = groups
#         print(A.shape, A)
#         Group_RASTER.append(A)
#     Group_RASTER=np.array(Group_RASTER)
#     print(Group_RASTER.shape)
#     Group_RASTER=np.sum(Group_RASTER,axis=0)
#     print(Group_RASTER.shape)
#     drv.write_img(Dir_output+"Group_"+str(year)+".tif", proj, geos,Group_RASTER.reshape(192,386))
#     SU_RASTER=np.array(SU_RASTER)
#     print(SU_RASTER.shape)
#     SU_RASTER=np.sum(SU_RASTER,axis=0)
#     print(SU_RASTER.shape)
#     drv.write_img(Dir_output+"SUi_"+str(year)+".tif", proj, geos,SU_RASTER.reshape(192,386))

"========================================= [2] Extracting cross-scale feature (CSFs) ========================================="
"==== (1) CSFs: dependent variable ===="
# Dir_raster="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Simple Spatialization/"
# Dir_output="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Features/Features_Y/"
# for yeari in range(1982,2016):
#     proj, geos, group = drv.read_img(Dir_raster + "Group_"+str(yeari)+".tif")
#     group = np.array(group).flatten()
#     group_unique_remove0 = np.unique(group)[np.unique(group) > 0]
#     proj, geos, SUi = drv.read_img(Dir_raster + "SUi_"+str(yeari)+".tif")
#     SUi = np.array(SUi).flatten()
#     DATA = group, SUi
#     DATA = np.array(DATA).transpose()
#
#     ID, MEAN_SUi = [], []
#     DATA_group, DATA_SUi = DATA[:, 0], DATA[:, 1]
#     for i in range(0, len(group_unique_remove0)):
#         group_ID = group_unique_remove0[i]
#         DATA_group_result = []
#         for j in range(0, group.shape[0]):
#             if DATA_group[j] == group_ID:
#                 DATA_group_result.append(1)
#             else:
#                 DATA_group_result.append(0)
#         DATA_group_result = np.array(DATA_group_result)
#         SUi_group_ID = DATA_group_result * DATA_SUi
#         SUi_group_ID = SUi_group_ID[SUi_group_ID != 0]
#         mean = np.mean(SUi_group_ID)
#         ID.append(group_ID)
#         MEAN_SUi.append(mean)
#     ID, MEAN_SUi = np.array(ID), np.array(MEAN_SUi)
#
#     DATA_result = ID, MEAN_SUi
#     DATA_result = np.array(DATA_result).transpose()
#     print(yeari,DATA_result.shape)
#     DataFrame(DATA_result).to_csv(Dir_output + "Y"+str(yeari)+".csv", sep=",", index=0)
# Dir_CountyLevel_CSVY = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Features/Features_Y/"
# csv_list = glob.glob(Dir_CountyLevel_CSVY +'*.csv')
# for i in csv_list:
#     fr = open(i, 'rb').read()
#     with open(Dir_CountyLevel_CSVY+'Result_Y.csv', 'ab') as f:   # Result_Y.csv is the CSFs: dependent variable
#         f.write(fr)

"==== (2) CSFs: independent variable===="
# Dir_mask = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Simple Spatialization/"
# Dir_raster = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 IndependentVariable/"
# Dir_output="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Features/Features_X/"
# variables=["1 Pasture suitability", "2 Distance to rivers", "3 Residential density", "4 Topographic range", "5 growing season NDVI",
#           "6 growing season PRE", "7 growing season TAS", "8 growing season RAD", "10 SoilMoisture", "13 PH_mean_layer1to3",
#           "14 AN_mean_layer1to3", "15 AP_mean_layer1to3","16 AK_mean_layer1to3", "17 protection level"]
# for i in range(0,len(variables)):
#     variable_name=variables[i]
#     variable_result=[]
#     for yeari in range(1982,2016):
#         proj, geos, group = drv.read_img(Dir_mask + "Group_"+str(yeari)+".tif")
#         group = np.array(group).flatten()
#         group_unique_remove0 = np.unique(group)[np.unique(group) > 0]
#         proj, geos, IMAGE = drv.read_img(Dir_raster + variable_name + "_"+str(yeari) + ".tif")
#         IMAGE = np.array(IMAGE).flatten()
#         DATA = group, IMAGE
#         DATA = np.array(DATA).transpose()
#         ID, MEAN_IMAGE = [], []
#         DATA_group, DATA_IMAGE = DATA[:, 0], DATA[:, 1]
#         for i in range(0, len(group_unique_remove0)):
#             group_ID = group_unique_remove0[i]
#             DATA_group_result = []
#             for j in range(0, group.shape[0]):
#                 if DATA_group[j] == group_ID:
#                     DATA_group_result.append(1)
#                 else:
#                     DATA_group_result.append(0)
#             DATA_group_result = np.array(DATA_group_result)
#             IMAGE_group_ID = DATA_group_result * DATA_IMAGE
#             IMAGE_group_ID = IMAGE_group_ID[IMAGE_group_ID != 0]
#             mean = np.mean(IMAGE_group_ID)
#             ID.append(group_ID)
#             MEAN_IMAGE.append(mean)
#         ID, MEAN_IMAGE = np.array(ID), np.array(MEAN_IMAGE)
#         DATA_result = ID, MEAN_IMAGE
#         DATA_result = np.array(DATA_result).transpose()
#         print(variable_name,yeari,DATA_result.shape)
#         DataFrame(DATA_result).to_csv(Dir_output + variable_name+"_"+str(yeari)+".csv", sep=",", index=0)
#
# Dir_CountyLevel_CSVX = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Features/Features_X/"
# variables = ["1 Pasture suitability", "2 Distance to rivers", "3 Residential density", "4 Topographic range", "5 growing season NDVI",
#              "6 growing season PRE", "7 growing season TAS", "8 growing season RAD", "10 SoilMoisture", "13 PH_mean_layer1to3",
#              "14 AN_mean_layer1to3", "15 AP_mean_layer1to3", "16 AK_mean_layer1to3", "17 protection level"]
# for i in range(0, len(variables)):
#     variable_name = variables[i]
#     csv_list = glob.glob(Dir_CountyLevel_CSVX +variable_name+"_"+'*.csv')
#     for i in csv_list:
#         fr = open(i, 'rb').read()
#         with open(Dir_CountyLevel_CSVX+"Result_"+variable_name+".csv", 'ab') as f:  # # Result_variable_name.csv is the CSFs: independent variable
#             f.write(fr)

"==== (3) CSFs: ALL ===="
# Dir_CountyLevel_CSVX =  "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Features/Features_X/"
# Dir_CountyLevel_CSVY= "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Features/Features_Y/"
# Dir_CountyLevel_CSVXY="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Improved Method/Features/"
# DATA_Y = pd.read_csv(Dir_CountyLevel_CSVY+"Result_Y.csv")
# DATA_1PastureSuitability = pd.read_csv(Dir_CountyLevel_CSVX+"Result_1 Pasture suitability.csv")
# DATA_2DistanceToRivers = pd.read_csv(Dir_CountyLevel_CSVX+"Result_2 Distance to rivers.csv")
# DATA_3ResidentialDensity = pd.read_csv(Dir_CountyLevel_CSVX+"Result_3 Residential density.csv")
# DATA_4TopographicRange = pd.read_csv(Dir_CountyLevel_CSVX+"Result_4 Topographic range.csv")
# DATA_5NDVI = pd.read_csv(Dir_CountyLevel_CSVX+"Result_5 growing season NDVI.csv")
# DATA_6PRE = pd.read_csv(Dir_CountyLevel_CSVX+"Result_6 growing season PRE.csv")
# DATA_7TAS = pd.read_csv(Dir_CountyLevel_CSVX+"Result_7 growing season TAS.csv")
# DATA_8RAD = pd.read_csv(Dir_CountyLevel_CSVX+"Result_8 growing season RAD.csv")
# DATA_10SoilMoisture = pd.read_csv(Dir_CountyLevel_CSVX+"Result_10 SoilMoisture.csv")
# DATA_13PH = pd.read_csv(Dir_CountyLevel_CSVX+"Result_13 PH_mean_layer1to3.csv")
# DATA_14AN = pd.read_csv(Dir_CountyLevel_CSVX+"Result_14 AN_mean_layer1to3.csv")
# DATA_15AP = pd.read_csv(Dir_CountyLevel_CSVX+"Result_15 AP_mean_layer1to3.csv")
# DATA_16AK = pd.read_csv(Dir_CountyLevel_CSVX+"Result_16 AK_mean_layer1to3.csv")
# DATA_17ProtectionLevel = pd.read_csv(Dir_CountyLevel_CSVX+"Result_17 protection level.csv")
# DATA0,DATA1,DATA2,DATA3,DATA4,DATA5,DATA6,DATA7,DATA8,DATA10,DATA13,DATA14,DATA15,DATA16,DATA17= \
#     np.array(DATA_Y)[:,1].reshape(-1,1),\
#     np.array(DATA_1PastureSuitability)[:,1].reshape(-1,1),np.array(DATA_2DistanceToRivers)[:,1].reshape(-1,1),np.array(DATA_3ResidentialDensity)[:,1].reshape(-1,1), \
#     np.array(DATA_4TopographicRange)[:,1].reshape(-1,1),np.array(DATA_5NDVI)[:,1].reshape(-1,1),np.array(DATA_6PRE)[:,1].reshape(-1,1),np.array(DATA_7TAS)[:,1].reshape(-1,1), \
#     np.array(DATA_8RAD)[:,1].reshape(-1,1),np.array(DATA_10SoilMoisture)[:,1].reshape(-1,1),np.array(DATA_13PH)[:,1].reshape(-1,1), np.array(DATA_14AN)[:,1].reshape(-1,1), \
#     np.array(DATA_15AP)[:,1].reshape(-1,1),np.array(DATA_16AK)[:,1].reshape(-1,1),np.array(DATA_17ProtectionLevel)[:,1].reshape(-1,1)
# DATA=np.concatenate((DATA0,DATA1,DATA2,DATA3,DATA4,DATA5,DATA6,DATA7,DATA8,DATA10,DATA13,DATA14,DATA15,DATA16,DATA17),axis=1)
# DATA=np.array(DATA)
# print(DATA.shape)
# DATA = DataFrame(DATA)
# DATA.to_csv(Dir_CountyLevel_CSVXY +"Features_YX.csv",sep=",",index=0)


"========================================= [3] Building RF model with partitioning ========================================="
# partitioning according to theoretical carrying capacity in excel
"=====(1) Overgrazing ===="
# DATA = pd.read_csv(r"C:/Users/mn/Desktop/Grazing spatialization/Features/Features_YX_Overgrazing.csv")
# DATA = DATA.fillna(0)
# x = DATA.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14]].values
# y = DATA.iloc[:, 0].values
# y = np.log(y+1) # transforming the response variable using natural log
# from sklearn.model_selection import train_test_split
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size =0.25, random_state = 1)
# print("train:",train_x.shape, train_y.shape,"test:",test_x.shape, test_y.shape)
# Model_RF = RandomForestRegressor()
# Model_RF.fit(train_x,train_y)
#
# Input_tifDir = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 IndependentVariable/"
# MASK_tifDir = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 RestrictedArea/"
# GRASS_tifDir = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 Grassland/"
# Dir_SU="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 DependentVariable/"
# Dir_SU_raster="C:/Users/mn/Desktop/Grazing spatialization/Simple Spatialization_SU per raster_Group/"
# output_tifDir = "C:/Users/mn/Desktop/Grazing spatialization/RF model/Overgrazing/"
# year_name_all=["1982","1983","1984","1985","1986","1987","1988","1989","1990","1991","1992",
#               "1993","1994","1995","1996","1997","1998","1999","2000","2001","2002","2003",
#               "2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015"]
# yeari=0
# while yeari in range(0,34):
#     yeari_name = year_name_all[yeari]
#     proj, geos, MASK = drv.read_img(MASK_tifDir+"Restrictedarea_"+yeari_name+".tif")
#     proj, geos, X1 = drv.read_img(Input_tifDir+"1 Pasture suitability_"+yeari_name+".tif")
#     proj, geos, X2 = drv.read_img(Input_tifDir+"2 Distance to rivers_"+yeari_name+".tif")
#     proj, geos, X3 = drv.read_img(Input_tifDir+"3 Residential density_"+yeari_name+".tif")
#     proj, geos, X4 = drv.read_img(Input_tifDir+"4 Topographic range_"+yeari_name+".tif")
#     proj, geos, X5 = drv.read_img(Input_tifDir+"5 growing season NDVI_"+yeari_name+".tif")
#     proj, geos, X6 = drv.read_img(Input_tifDir+"6 growing season PRE_"+yeari_name+".tif")
#     proj, geos, X7 = drv.read_img(Input_tifDir+"7 growing season TAS_"+yeari_name+".tif")
#     proj, geos, X8 = drv.read_img(Input_tifDir+"8 growing season RAD_"+yeari_name+".tif")
#     proj, geos, X10 = drv.read_img(Input_tifDir+"10 SoilMoisture_"+yeari_name+".tif")
#     proj, geos, X13 = drv.read_img(Input_tifDir+"13 PH_mean_layer1to3_"+yeari_name+".tif")
#     proj, geos, X14 = drv.read_img(Input_tifDir+"14 AN_mean_layer1to3_"+yeari_name+".tif")
#     proj, geos, X15 = drv.read_img(Input_tifDir+"15 AP_mean_layer1to3_"+yeari_name+".tif")
#     proj, geos, X16 = drv.read_img(Input_tifDir+"16 AK_mean_layer1to3_"+yeari_name+".tif")
#     proj, geos, X17 = drv.read_img(Input_tifDir+"17 protection level_"+yeari_name+".tif")
#     proj, geos, X18 = drv.read_img(Input_tifDir+"18 D_RF_"+yeari_name+".tif")
#     data=np.array(X1).flatten(),np.array(X2).flatten(),np.array(X3).flatten(),np.array(X4).flatten(),\
#          np.array(X5).flatten(),np.array(X6).flatten(),np.array(X7).flatten(),np.array(X8).flatten(),\
#          np.array(X10).flatten(),np.array(X13).flatten(),np.array(X14).flatten(),np.array(X15).flatten(),\
#          np.array(X16).flatten(), np.array(X17).flatten(),np.array(X18).flatten()
#     print(np.array(data).shape)
#     Raster_dataX=pd.DataFrame(data).transpose()
#     Raster_dataX = Raster_dataX.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
#     proj, geos, GRASS = drv.read_img(GRASS_tifDir+"Grassland.tif")
#     GRASS=np.array(GRASS)
#     proj, geos, Overgrazing = drv.read_img(Dir_SU_raster + "Group" + yeari_name + ".tif")   # 超载，传统方法
#     Overgrazing=np.array(Overgrazing)
#     Overgrazing[Overgrazing<8000]=0
#     Overgrazing[Overgrazing != 0] = 1
#     Model_RF_predictY=Model_RF.predict(Raster_dataX)
#     Model_RF_predictY_rasters=[]
#     for rasteri in range(0,GRASS.shape[0]*GRASS.shape[1]):
#         Model_RF_predictY_rasteri=np.exp(Model_RF_predictY[rasteri])
#         Model_RF_predictY_rasters.append(Model_RF_predictY_rasteri)
#     Model_RF_predictY = np.array(Model_RF_predictY_rasters).reshape(192,386)
#     Model_RF_predictY = Model_RF_predictY * MASK
#     Model_RF_predictY[Model_RF_predictY == 0] = -1
#     Model_RF_predictY=Model_RF_predictY + GRASS
#     Model_RF_predictY[Model_RF_predictY < 0] = -1
#     Model_RF_predictY=Model_RF_predictY*Overgrazing
#     drv.write_img(output_tifDir + "RF_" + yeari_name + ".tif", proj, geos, Model_RF_predictY)
#     print(yeari_name,"RF")
#     yeari+=1

"=====(2) Non-overgrazing ===="
# DATA = pd.read_csv(r"C:/Users/mn/Desktop/Grazing spatialization/Features/Features_YX_未超载.csv")
# DATA = DATA.fillna(0)
# x = DATA.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14]].values           # x1-X7,
# y = DATA.iloc[:, 0].values
# y = np.log(y+1)
# from sklearn.model_selection import train_test_split
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size =0.25, random_state = 1)
# print("train:",train_x.shape, train_y.shape,"test:",test_x.shape, test_y.shape)
# Model_RF = RandomForestRegressor()
# Model_RF.fit(train_x,train_y)
# Input_tifDir = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 IndependentVariable/"
# MASK_tifDir = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 RestrictedArea/"
# GRASS_tifDir = "C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 Grassland/"
# Dir_SU="C:/Users/mn/Desktop/Grazing spatialization/MODEL_Data/0 DependentVariable/"
# Dir_SU_raster="C:/Users/mn/Desktop/Grazing spatialization/Simple Spatialization_SU per raster_Group/"
# output_tifDir = "C:/Users/mn/Desktop/Grazing spatialization/RF model/Non-overgrazing/"
#
# year_name_all=["1982","1983","1984","1985","1986","1987","1988","1989","1990","1991","1992",
#                "1993","1994","1995","1996","1997","1998","1999","2000","2001","2002","2003",
#                "2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015"]
# yeari=0
# while yeari in range(0,34):
#     yeari_name = year_name_all[yeari]
#     proj, geos, MASK = drv.read_img(MASK_tifDir+"Restrictedarea_"+yeari_name+".tif")
#     proj, geos, X1 = drv.read_img(Input_tifDir+"1 Pasture suitability_"+yeari_name+".tif")
#     proj, geos, X2 = drv.read_img(Input_tifDir+"2 Distance to rivers_"+yeari_name+".tif")
#     proj, geos, X3 = drv.read_img(Input_tifDir+"3 Residential density_"+yeari_name+".tif")
#     proj, geos, X4 = drv.read_img(Input_tifDir+"4 Topographic range_"+yeari_name+".tif")
#     proj, geos, X5 = drv.read_img(Input_tifDir+"5 growing season NDVI_"+yeari_name+".tif")
#     proj, geos, X6 = drv.read_img(Input_tifDir+"6 growing season PRE_"+yeari_name+".tif")
#     proj, geos, X7 = drv.read_img(Input_tifDir+"7 growing season TAS_"+yeari_name+".tif")
#     proj, geos, X8 = drv.read_img(Input_tifDir+"8 growing season RAD_"+yeari_name+".tif")
#     proj, geos, X10 = drv.read_img(Input_tifDir+"10 SoilMoisture_"+yeari_name+".tif")
#     proj, geos, X13 = drv.read_img(Input_tifDir+"13 PH_mean_layer1to3_"+yeari_name+".tif")
#     proj, geos, X14 = drv.read_img(Input_tifDir+"14 AN_mean_layer1to3_"+yeari_name+".tif")
#     proj, geos, X15 = drv.read_img(Input_tifDir+"15 AP_mean_layer1to3_"+yeari_name+".tif")
#     proj, geos, X16 = drv.read_img(Input_tifDir+"16 AK_mean_layer1to3_"+yeari_name+".tif")
#     proj, geos, X17 = drv.read_img(Input_tifDir+"17 protection level_"+yeari_name+".tif")
#     proj, geos, X18 = drv.read_img(Input_tifDir+"18 D_RF_"+yeari_name+".tif")
#     data=np.array(X1).flatten(),np.array(X2).flatten(),np.array(X3).flatten(),np.array(X4).flatten(), \
#          np.array(X5).flatten(),np.array(X6).flatten(),np.array(X7).flatten(),np.array(X8).flatten(), \
#          np.array(X10).flatten(),np.array(X13).flatten(),np.array(X14).flatten(),np.array(X15).flatten(), \
#          np.array(X16).flatten(), np.array(X17).flatten(),np.array(X18).flatten()
#     print(np.array(data).shape)
#     Raster_dataX=pd.DataFrame(data).transpose()
#     Raster_dataX = Raster_dataX.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
#     proj, geos, GRASS = drv.read_img(GRASS_tifDir+"Grassland.tif")
#     GRASS=np.array(GRASS)
#     proj, geos, Non_overgrazing = drv.read_img(Dir_SU_raster + "Group" + yeari_name + ".tif")   # 未超载，传统方法
#     Non_overgrazing=np.array(Non_overgrazing)
#     Non_overgrazing[Non_overgrazing<8000]=1
#     Non_overgrazing[Non_overgrazing != 1] = 0
#
#     Model_RF_predictY=Model_RF.predict(Raster_dataX)
#     Model_RF_predictY_rasters=[]
#     for rasteri in range(0,GRASS.shape[0]*GRASS.shape[1]):
#         Model_RF_predictY_rasteri=np.exp(Model_RF_predictY[rasteri])
#         Model_RF_predictY_rasters.append(Model_RF_predictY_rasteri)
#     Model_RF_predictY = np.array(Model_RF_predictY_rasters).reshape(192,386)
#     Model_RF_predictY = Model_RF_predictY * MASK
#     Model_RF_predictY[Model_RF_predictY == 0] = -1
#     Model_RF_predictY=Model_RF_predictY + GRASS
#     Model_RF_predictY[Model_RF_predictY < 0] = -1
#     Model_RF_predictY = Model_RF_predictY * Non_overgrazing
#     drv.write_img(output_tifDir + "RF_" + yeari_name + ".tif", proj, geos, Model_RF_predictY)
#     print(yeari_name,"RF")
#     yeari+=1

"=====(3)Merging the above two parts ==============="
# input_overgrazing = "C:/Users/mn/Desktop/Grazing spatialization/RF model/Overgrazing"
# input_nonovergrazing = "C:/Users/mn/Desktop/Grazing spatialization/RF model/Non-overgrazing/"
# output = "C:/Users/mn/Desktop/Grazing spatialization/RF model/"
# year_name_all=["1982","1983","1984","1985","1986","1987","1988","1989","1990","1991","1992",
#               "1993","1994","1995","1996","1997","1998","1999","2000","2001","2002","2003",
#               "2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015"]
# yeari=0
# while yeari in range(0,34):
#     yeari_name = year_name_all[yeari]
#     proj, geos, nonovergrazing = drv.read_img(input_nonovergrazing  + "RF_" + yeari_name + ".tif")  # 未超载
#     proj, geos, overgrazing = drv.read_img(input_overgrazing  + "RF_" + yeari_name + ".tif")  # 未超载
#     nonovergrazing,overgrazing = np.array(nonovergrazing),np.array(overgrazing)
#     result=nonovergrazing+overgrazing
#     drv.write_img(output+"Imporved_RF_"+yeari_name+".tif", proj, geos,result)
#     print(yeari_name,"rf")
#     yeari+=1


"========================================= [4] Correcting residuals of dataset ========================================="
# input_image=r"C:/Users/mn/Desktop/Grazing spatialization/RF model/"
# input_group=r"C:/Users/mn/Desktop/Grazing spatialization/Simple Spatialization/"
# input_SUi=r"C:/Users/mn/Desktop/Grazing spatialization/Simple Spatialization/"
# output_CSV=r"C:/Users/mn/Desktop/Grazing spatialization/SD/"
# model="RF"
#
# for yeari in range(1982,2016):
#     yeari_name = yeari
#     print(yeari_name)
#     proj, geos, IMAGE = drv.read_img(input_image+"Imporved_"+model+"_"+str(yeari_name)+".tif")
#     proj, geos, GROUP = drv.read_img(input_group+"Group_"+str(yeari_name)+".tif")
#     proj, geos, SUi= drv.read_img(input_group+"SUi_"+str(yeari_name)+".tif")
#     IMAGE,GROUP,SUi=np.array(IMAGE).flatten(),np.array(GROUP).flatten(),np.array(SUi).flatten()
#     group_unique_remove0 = np.unique(GROUP)[np.unique(GROUP) > 0]
#     DATA = GROUP,IMAGE,SUi
#     DATA = np.array(DATA).transpose()
#     print(DATA.shape)
#     ID, MEAN_IMAGE,MEAN_SUi,SD = [], [], [], []
#     DATA_GROUP,DATA_IMAGE, DATA_SUi = DATA[:, 0], DATA[:, 1], DATA[:, 2]
#     for i in range(0, len(group_unique_remove0)):
#         group_ID = group_unique_remove0[i]
#         ID.append(group_ID)
#         DATA_group_result = []
#         for j in range(0, GROUP.shape[0]):
#             if DATA_GROUP[j] == group_ID:
#                 DATA_group_result.append(1)
#             else:
#                 DATA_group_result.append(0)
#         DATA_group_result = np.array(DATA_group_result)
#         SUi_group_ID = DATA_group_result * DATA_SUi
#         SUi_group_ID = SUi_group_ID[SUi_group_ID != 0]
#         mean__SUi = np.mean(SUi_group_ID)
#         MEAN_SUi.append(mean__SUi)
#         IMAGE_group_ID = DATA_group_result * DATA_IMAGE
#         IMAGE_group_ID = IMAGE_group_ID[IMAGE_group_ID != 0]
#         mean__IMAGE = np.mean(IMAGE_group_ID)
#         MEAN_IMAGE.append(mean__IMAGE)
#         MEAN_sd=mean__SUi-mean__IMAGE
#         SD.append(MEAN_sd)
#     ID, MEAN_IMAGE,MEAN_SUi,SD = np.array(ID), np.array(MEAN_IMAGE),np.array(MEAN_SUi), np.array(SD)
#     DATA_result = ID, MEAN_IMAGE,MEAN_SUi,SD
#     DATA_result = np.array(DATA_result).transpose()
#     print(yeari,DATA_result.shape)
#     DataFrame(DATA_result).to_csv(output_CSV +model+ "_SD_"+str(yeari)+".csv", sep=",", index=0)
#
# input_group=r"C:/Users/mn/Desktop/Grazing spatialization/Simple Spatialization/"
# input_CSV=r"C:/Users/mn/Desktop/Grazing spatialization/SD/"
# output_image=r"C:/Users/mn/Desktop/Grazing spatialization/SD/"
# model="RF"
# for yeari in range(1982,2016):
#     yeari_name = yeari
#     print(yeari_name)
#     proj, geos, GROUP = drv.read_img(input_group+"Group_"+str(yeari_name)+".tif")
#     GROUP=np.array(GROUP).flatten()
#     SD = pd.read_csv(input_CSV+model+"_SD_"+str(yeari)+".csv")
#     SD=np.array(SD)
#     lines=SD.shape[0]
#     print(lines)
#     result=[]
#     for linei in range(0,len(GROUP)):
#         GROUP_linei=GROUP[linei]
#         for idi in range(0,lines):
#             SD0,SD3=SD[idi,0],SD[idi,3]
#             if GROUP_linei==SD0:
#                 GROUP_linei=SD3
#             # else:
#             #     GROUP_linei=0
#         result.append(GROUP_linei)
#     result=np.array(result).reshape(192,386)
#     drv.write_img(output_image+model+"_"+str(yeari_name)+"_SD.tif", proj, geos, result)
#
# input_SD = r"C:/Users/mn/Desktop/Grazing spatialization/SD/"
# input_image = r"C:/Users/mn/Desktop/Grazing spatialization/RF model/"
# input_mask = r"C:/Users/mn/Desktop/Grazing spatialization/0 RestrictedArea/"
# output_image = r"C:/Users/mn/Desktop/Grazing spatialization/Revised RF_model/"
# model = "RF"
# for yeari in range(1982,2016):
#     yeari_name = yeari
#     print(yeari_name)
#     proj, geos, SD = drv.read_img(input_SD+model+"_"+str(yeari_name)+"_SD.tif")
#     proj, geos, image = drv.read_img(input_image+"Imporved_"+model+"_"+str(yeari_name)+".tif")
#     proj, geos, mask = drv.read_img(input_mask+"Restrictedarea_"+str(yeari_name)+".tif")
#     SD,IMAGE,mask = np.array(SD),np.array(image),np.array(mask)
#     RESULT=(SD+IMAGE)*mask
#     RESULT[RESULT<-1]=0
#     RESULT[RESULT>10000000000]=-1
#     drv.write_img(output_image+model+"_"+str(yeari_name)+"_modify.tif", proj, geos, RESULT)