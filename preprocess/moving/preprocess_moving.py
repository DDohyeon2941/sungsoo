# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:50:59 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np

temp_df = pd.read_csv(r'sungsoo_grid_arrival_2024.csv')
temp_df1 = pd.read_csv(r'..\sales\sungsoo_2nd_drop_polygons_45.csv', index_col=0)




temp_df['date'] = pd.to_datetime(temp_df[['year', 'month', 'day']])
temp_df.drop(columns=['year','month','day'], inplace=True)


temp_df.loc[:, 'date'] = pd.to_datetime(temp_df['date'])

temp_df = temp_df.loc[temp_df['d_cell_id'].isin(temp_df1['1'])].reset_index(drop=True)

grid_dic = dict(zip(temp_df1['1'],temp_df1['0']))

temp_df.loc[:, 'd_cell_id'] = [grid_dic[xx] for xx in temp_df['d_cell_id']]

temp_df[['date','d_cell_id','avg_move_dist', 'avg_move_time',
       'sum_male_00_19', 'sum_male_20_29', 'sum_male_30_39', 'sum_male_40_49',
       'sum_male_50_59', 'sum_male_60_69', 'sum_male_70_79', 'sum_male_80_89',
       'sum_feml_00_19', 'sum_feml_20_29', 'sum_feml_30_39', 'sum_feml_40_49',
       'sum_feml_50_59', 'sum_feml_60_69', 'sum_feml_70_79', 'sum_feml_80_89',
       'sum_total_cnt']].groupby(['date','d_cell_id']).mean().to_csv(r'sungsoo_grid_arrival_2024_20241114.csv')



temp_df.groupby(['date','d_cell_id']).mean().unstack(fill_value=0).stack().to_csv(r'sungsoo_grid_arrival_2023_20241114.csv')
