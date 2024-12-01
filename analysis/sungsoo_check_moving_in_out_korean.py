# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:10:07 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

temp_df2 = pd.read_csv(r'moving_in_out_sex_age.csv')

temp_df1 = pd.read_csv(r'sungsoo_2nd_drop_polygons_45.csv', index_col=0)

grid_dic = dict(zip(temp_df1['1'],temp_df1['0']))

new_temp_df2 = temp_df2.loc[temp_df2['o_cell_id'].isin(temp_df1['1'])].reset_index(drop=True)

new_temp_df2['date'] = pd.to_datetime(new_temp_df2[['year', 'month', 'day']])
new_temp_df2.drop(columns=['year','month','day'], inplace=True)
new_temp_df2.loc[:,'o_cell_id'] =  [grid_dic[xx] for xx in new_temp_df2['o_cell_id']]

keep_grid = ['다사60aa49bb' # 뚝섬역
                , '다사60ba49ab' # 가죽거리
                ]

paeup_grid = ['다사60ba49bb' # 성결교회
                , '다사60ba48bb' # 뚝도시장
                ]


grid_color_mapping = {
    '다사60aa49bb': 'skyblue',   # 뚝섬역
    '다사60ba49ab': 'blue',   # 가죽거리
    '다사60ba49bb': 'orange',    # 성결교회
    '다사60ba48bb': 'red', # 뚝도시장
}

paeup_color_mapping = {
    0: 'blue',   #  지속
    1: 'red',   # 폐업
}
target_variable = 'sum_total_cnt'
gr_df2 = new_temp_df2.groupby(['o_cell_id','date','move_type']).mean()[target_variable].unstack()

#print(gr_df2)
#print(gr_df2.columns)
print(gr_df2.loc[keep_grid])
#print(gr_df2.loc['다사60aa49bb'].sum(axis=1).values)
#print(gr_df2.loc['다사60aa49bb'].sum(axis=1).values.reshape(-1,1))
#print((gr_df2.loc['다사60aa49bb'].values / gr_df2.loc['다사60aa49bb'].sum(axis=1).values.reshape(-1,1))[:,0])

#%% example
fig1, axes1 = plt.subplots(1,1, figsize=(16,6))

#print(gr_df2.loc[keep_grid].index.duplicated().sum())  

axes1.set_xticks(np.arange(0,517,100))
axes1.set_xticklabels(gr_df2.loc['다사60aa49bb'].index[np.arange(0,517,100)].date, rotation=90)

#axes1.plot((gr_df2.loc[keep_grid].values / gr_df2.loc[keep_grid].sum(axis=1).values.reshape(-1,1))[:,1], label='keep', color= paeup_color_mapping[0])
#axes1.plot((gr_df2.loc[paeup_grid].values / gr_df2.loc[paeup_grid].sum(axis=1).values.reshape(-1,1))[:,1], label='paeup', color= paeup_color_mapping[1])

axes1.plot((gr_df2.loc['다사60aa49bb'].values / gr_df2.loc['다사60aa49bb'].sum(axis=1).values.reshape(-1,1))[:,1], label='뚝섬역', color= grid_color_mapping['다사60aa49bb'])
axes1.plot((gr_df2.loc['다사60ba49ab'].values / gr_df2.loc['다사60ba49ab'].sum(axis=1).values.reshape(-1,1))[:,1], label='가죽거리', color= grid_color_mapping['다사60ba49ab'])
axes1.plot((gr_df2.loc['다사60ba49bb'].values / gr_df2.loc['다사60ba49bb'].sum(axis=1).values.reshape(-1,1))[:,1], label='성결교회', color= grid_color_mapping['다사60ba49bb'])
axes1.plot((gr_df2.loc['다사60ba48bb'].values / gr_df2.loc['다사60ba48bb'].sum(axis=1).values.reshape(-1,1))[:,1], label='뚝도시장', color= grid_color_mapping['다사60ba48bb'])

axes1.set_ylabel('in ratio')

axes1.legend()
#plt.savefig(f'images/efflux_ratio_{target_variable}.png')
#plt.show()
#plt.close()