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


gr_df2 = new_temp_df2.groupby(['o_cell_id','date','move_type']).mean()['sum_feml_20_29'].unstack()

#%% example
fig1, axes1 = plt.subplots(1,1, figsize=(20,6))

axes1.plot((gr_df2.loc['다사60ab49ab'].values / gr_df2.loc['다사60ab49ab'].sum(axis=1).values.reshape(-1,1))[:,0], label='paeup')
axes1.plot((gr_df2.loc['다사59bb49ba'].values / gr_df2.loc['다사59bb49ba'].sum(axis=1).values.reshape(-1,1))[:,0], label='keep')
axes1.set_ylabel('in ratio')

axes1.legend()
