# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:11:51 2022

@author: 309
"""

x = 282509.61179709
y = 2695809.5982648
nrows = 9441
cellsize = 5
row_col = joblib.load(r"E:\宜蘭崩塌\asc_5m\5m_studyarea_row_col.pkl")
row_col = row_col.to_numpy()


xxx_p1 = (x+(row_col_p1[:,1]+1)*cellsize)-1
yyy_p1 = (y+(nrows-row_col_p1[:,0]-1)*cellsize)+1
   

trans_LS = {'x':xxx_p1,'y': yyy_p1}
df_all = pd.DataFrame(data=trans_LS)
df_all['L']= ppp
df_all.to_csv(r"E:\宜蘭崩塌\asc_5m\5m_studyarea_2015_picture/2015draw_%s.txt"%(iii), index=None, sep=' ')

#%%
x = 282510.84524418
y = 2695812.7721455
nrows = 1181
cellsize = 40
row_col = joblib.load(r"E:\宜蘭崩塌\asc_40\40m_all_row_col.pkl")
row_col = row_col.to_numpy()

xxx_p1 = (x+(row_col[:,1]+1)*cellsize)-20
yyy_p1 = (y+(nrows-row_col[:,0]-1)*cellsize)+20
   

trans_LS = {'x':xxx_p1,'y': yyy_p1}
df_all = pd.DataFrame(data=trans_LS)
df_all['L']= ppp
df_all.to_csv(r"E:\宜蘭崩塌\asc_40\40m_studyarea_picture/2017draw1.txt", index=None, sep=' ')