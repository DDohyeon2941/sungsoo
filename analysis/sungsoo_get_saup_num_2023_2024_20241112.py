# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:20:23 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

temp_df = pd.read_csv(r'sungsoo_paeup_grid_coef_20240920.csv')

new_temp_df = temp_df.fillna('2025-01-01')



(new_temp_df.loc[new_temp_df['end_year']>2022].groupby(['grid1','end_year','cate_mask']).count()['paeup'].unstack(fill_value=0)[[0,1,2]].unstack()).to_csv(r'saup_number.csv')



#%%


new_temp_df.loc[:,'end_date'] = pd.to_datetime(new_temp_df['end_date'])
new_temp_df.loc[:, 'end_month'] = new_temp_df['end_date'].dt.month
new_temp_df.loc[:, 'end_year'] = new_temp_df['end_date'].dt.year

new_temp_df.loc[:, 'start_date']= pd.to_datetime(new_temp_df['start_date'])
new_temp_df.loc[:, 'start_year']= new_temp_df['start_date'].dt.year
new_temp_df.loc[:, 'start_year_mask']= [2022 if xx < 2023 else xx for xx in new_temp_df['start_year']]

test_df = new_temp_df.loc[new_temp_df['end_year']>2022].groupby(['grid1','end_year','end_month','cate_mask']).count()['paeup'].unstack(fill_value=0)[[0,1,2]].unstack()[2].fillna(0).stack().unstack([-2,-1], fill_value=0)

#%%
new_temp_df.loc[new_temp_df['end_year']>2022].groupby(['grid1','end_year','cate_mask']).count()['paeup'].unstack()[[0,1,2]].unstack()[2]


new_temp_df.loc[new_temp_df['end_year']>2022].groupby(['grid1','start_year_mask','cate_mask']).count()['paeup'].unstack()[[0,1,2]].unstack()[2]



































