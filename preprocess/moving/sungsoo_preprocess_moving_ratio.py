# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:50:59 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np

temp_df = pd.read_csv(r'sungsoo_grid_arrival_2024.csv')
temp_df1 = pd.read_csv(r'sungsoo_2nd_drop_polygons_45.csv', index_col=0)
temp_df2 = pd.read_csv(r'성동구_격자.csv')



temp_df['date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])
temp_df.drop(columns=['year','month','day'], inplace=True)


t1=temp_df.loc[temp_df['o_cell_id'].isin(temp_df2['cell_id'])].groupby(['date','d_cell_id']).sum()
t2=temp_df.groupby(['date','d_cell_id']).sum()


t3 = (1-(t1/t2)).reset_index().fillna(0)

grid_dic = dict(zip(temp_df1['1'],temp_df1['0']))


t3 = t3.loc[t3['d_cell_id'].isin(temp_df1['1'])].reset_index(drop=True)
t3.loc[:,'d_cell_id'] =  [grid_dic[xx] for xx in t3['d_cell_id']]

t3.columns = ['date', 'd_cell_id', 'avg_move_dist1', 'avg_move_time1', 'sum_male_00_191',
       'sum_male_20_291', 'sum_male_30_391', 'sum_male_40_491', 'sum_male_50_591',
       'sum_male_60_691', 'sum_male_70_791', 'sum_male_80_891', 'sum_feml_00_191',
       'sum_feml_20_291', 'sum_feml_30_391', 'sum_feml_40_491', 'sum_feml_50_591',
       'sum_feml_60_691', 'sum_feml_70_791', 'sum_feml_80_891', 'sum_total_cnt1']

t3.groupby(['date','d_cell_id']).mean().unstack(fill_value=0).stack().to_csv(r'sungsoo_grid_arrival_2024_20241125.csv')



#%%

temp_df = pd.read_csv(r'sungsoo_grid_arrival_2023.csv')
temp_df1 = pd.read_csv(r'sungsoo_2nd_drop_polygons_45.csv', index_col=0)
temp_df2 = pd.read_csv(r'성동구_격자.csv')


temp_df['date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])
temp_df.drop(columns=['year','month','day'], inplace=True)


t1=temp_df.loc[temp_df['o_cell_id'].isin(temp_df2['cell_id'])].groupby(['date','d_cell_id']).sum()
t2=temp_df.groupby(['date','d_cell_id']).sum()


t3 = (1-(t1/t2)).reset_index().fillna(0)

grid_dic = dict(zip(temp_df1['1'],temp_df1['0']))


t3 = t3.loc[t3['d_cell_id'].isin(temp_df1['1'])].reset_index(drop=True)
t3.loc[:,'d_cell_id'] =  [grid_dic[xx] for xx in t3['d_cell_id']]

t3.columns = ['date', 'd_cell_id', 'avg_move_dist1', 'avg_move_time1', 'sum_male_00_191',
       'sum_male_20_291', 'sum_male_30_391', 'sum_male_40_491', 'sum_male_50_591',
       'sum_male_60_691', 'sum_male_70_791', 'sum_male_80_891', 'sum_feml_00_191',
       'sum_feml_20_291', 'sum_feml_30_391', 'sum_feml_40_491', 'sum_feml_50_591',
       'sum_feml_60_691', 'sum_feml_70_791', 'sum_feml_80_891', 'sum_total_cnt1']

t3.groupby(['date','d_cell_id']).mean().unstack(fill_value=0).stack().to_csv(r'sungsoo_grid_arrival_2023_20241125.csv')
