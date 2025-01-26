# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:20:23 2024

@author: dohyeon
"""

import pandas as pd


temp_df = pd.read_csv(r'sungsoo_paeup_grid_coef_20240920.csv')

new_temp_df = temp_df.fillna(2025)



(new_temp_df.loc[new_temp_df['end_year']>2022].groupby(['grid1','end_year','cate_mask']).count()['paeup'].unstack(fill_value=0)[[0,1,2]].unstack()).to_csv(r'saup_number.csv')
























