# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:24:46 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np
import ast

temp_df = pd.read_csv(r'combined_transport_distance.csv')



temp_df['bike_distances'] = temp_df['bike_distances'].apply(ast.literal_eval)
temp_df['bus_distances'] = temp_df['bus_distances'].apply(ast.literal_eval)
temp_df['subway_distances'] = temp_df['subway_distances'].apply(ast.literal_eval)



bike_dist_df = pd.DataFrame(data=dict(zip(temp_df['0'], temp_df['bike_distances']))).T
bus_dist_df = pd.DataFrame(data=dict(zip(temp_df['0'], temp_df['bus_distances']))).T
subway_dist_df = pd.DataFrame(data=dict(zip(temp_df['0'], temp_df['subway_distances']))).T




bike_weight_df = 1-(bike_dist_df / bike_dist_df.max(axis=0))
bus_weight_df = 1-(bus_dist_df / bus_dist_df.max(axis=0))
subway_weight_df = 1-(subway_dist_df / subway_dist_df.max(axis=0))


#%%

import seaborn as sns
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)



sns.heatmap(subway_weight_df)
sns.heatmap(bus_weight_df)
sns.heatmap(bike_weight_df)


#%%

bike_weight_df.to_csv('bike_weight.csv')
bus_weight_df.to_csv('bus_weight.csv')
subway_weight_df.to_csv('subway_weight.csv')
