# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:50:04 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np

"""공유자전거"""
bike_weight_df = pd.read_csv(r'bike_weight.csv', index_col=0)
bike_weight_df.drop(columns = 'ST-106', inplace=True)
#bike_weight_df.to_csv('bike_weight.csv')

temp_df = pd.read_csv(r'bike\seongsu_bike_2023.csv')
temp_df1 = pd.read_csv(r'bike\seongsu_bike_2024.csv').dropna()


new_temp_df = temp_df.loc[temp_df['return_st_id'].isin(bike_weight_df.columns)].reset_index(drop=True)
new_temp_df1 = temp_df1.loc[temp_df1['return_st_id'].isin(bike_weight_df.columns)].reset_index(drop=True)

new_temp_df.loc[:, 'return_hour'] = [int(xx[:2]) if  len(xx)==4 else int(xx.zfill(4)[:2]) for xx in new_temp_df['return_hour'].astype(int).astype(str)]

new_temp_df1.loc[:, 'return_hour'] = [int(xx[:2]) if  len(xx)==4 else int(xx.zfill(4)[:2]) for xx in new_temp_df1['return_hour'].astype(int).astype(str)]


bike_usage_2023 = new_temp_df.groupby(['date','return_st_id','return_hour']).sum()['trip_count'].unstack(fill_value=0)[np.arange(24)].mean(axis=1).unstack(fill_value=0)
bike_usage_2024 = new_temp_df1.groupby(['date','return_st_id','return_hour']).sum()['trip_count'].unstack(fill_value=0)[np.arange(24)].mean(axis=1).unstack(fill_value=0)



bike_weight_df_2023 = bike_weight_df[bike_usage_2023.columns]
bike_weight_df_2024 = bike_weight_df[bike_usage_2024.columns]


bike_weighted_df_2023 = pd.concat([(bike_usage_2023 * bike_weight_df_2023.iloc[xx]).sum(axis=1) / bike_weight_df_2023.iloc[xx].sum() for xx in np.arange(len(bike_weight_df_2023))], axis=1)

bike_weighted_df_2023.columns = bike_weight_df_2023.index



bike_weighted_df_2024= pd.concat([(bike_usage_2024 * bike_weight_df_2024.iloc[xx]).sum(axis=1) / bike_weight_df_2024.iloc[xx].sum() for xx in np.arange(len(bike_weight_df_2024))], axis=1)

bike_weighted_df_2024.columns = bike_weight_df_2024.index


bike_total_df = pd.concat([bike_weighted_df_2023, bike_weighted_df_2024])


#bike_total_df.to_csv(r'sungsoo_bike_weighted_usage.csv')

#%%
"""버스"""

bus_weight_df = pd.read_csv(r'bus_weight.csv', index_col=0)

bus_temp_df = pd.read_csv(r'bus\seoungsu_card_bus_data.csv')



new_bus_temp_df = bus_temp_df.loc[bus_temp_df['ARS_ID'].isin(bus_weight_df.columns.astype(int))].reset_index(drop=True)


bus_weight_df = bus_weight_df[new_bus_temp_df['ARS_ID'].unique().astype(str)]
bus_weight_df.columns = bus_weight_df.columns.astype(int)

bus_board_df = new_bus_temp_df.groupby(['use_date','ARS_ID']).mean().unstack(fill_value=0).stack()['board_cnt'].unstack(fill_value=0)


bus_resembark_df = new_bus_temp_df.groupby(['use_date','ARS_ID']).mean().unstack(fill_value=0).stack()['disembark_cnt'].unstack(fill_value=0)


bus_weighted_df = pd.concat([(bus_board_df * bus_weight_df.iloc[xx] ).sum(axis=1) / bus_weight_df.iloc[xx].sum() for xx in np.arange(len(bus_weight_df))], axis=1)

bus_weighted_df.columns = bus_weight_df.index

###
bus_weighted_df1 = pd.concat([(bus_resembark_df * bus_weight_df.iloc[xx] ).sum(axis=1) / bus_weight_df.iloc[xx].sum() for xx in np.arange(len(bus_weight_df))], axis=1)

bus_weighted_df.columns = bus_weight_df.index
bus_weighted_df1.columns = bus_weight_df.index

#bus_weighted_df.to_csv(r'sungsoo_bus_weighted_board.csv')
#bus_weighted_df1.to_csv(r'sungsoo_bus_weighted_resembark.csv')
#%%
"""지하철"""
subway_weight_df = pd.read_csv(r'subway_weight.csv', index_col=0)

subway_temp_df = pd.read_csv(r'station\seongsu_card_subway_data.csv')



subway_board_df = subway_temp_df[['use_date','st_nm','board_cnt']].groupby(['use_date','st_nm']).mean()['board_cnt'].unstack()

subway_resembark_df = subway_temp_df[['use_date','st_nm','disembark_cnt']].groupby(['use_date','st_nm']).mean()['disembark_cnt'].unstack()



subway_weighted_df = pd.concat([(subway_board_df * subway_weight_df.iloc[xx] ).sum(axis=1) / subway_weight_df.iloc[xx].sum() for xx in np.arange(len(subway_weight_df))], axis=1)

subway_weighted_df.columns = subway_weight_df.index

##
subway_weighted_df1 = pd.concat([(subway_resembark_df * subway_weight_df.iloc[xx] ).sum(axis=1) / subway_weight_df.iloc[xx].sum() for xx in np.arange(len(subway_weight_df))], axis=1)

subway_weighted_df1.columns = subway_weight_df.index


##

#subway_weighted_df.to_csv(r'sungsoo_subway_weighted_board.csv')
#subway_weighted_df1.to_csv(r'sungsoo_subway_weighted_resembark.csv')


#%% 결합

def convert_and_filtering(whole_df):
    whole_df.index= pd.to_datetime(whole_df.index.astype(str))
    return whole_df.loc[:'2024-04-30']


bike_new_total_df = pd.DataFrame(index=bus_weighted_df.index, columns=bus_weighted_df.columns)
bike_new_total_df.loc[bike_total_df.index, bike_total_df.columns] = bike_total_df.values
bike_new_total_df.fillna(0, inplace=True)

bike_new_total_df = convert_and_filtering(bike_new_total_df)
bus_weighted_df = convert_and_filtering(bus_weighted_df)
bus_weighted_df1 = convert_and_filtering(bus_weighted_df1)

subway_weighted_df = convert_and_filtering(subway_weighted_df)
subway_weighted_df1 = convert_and_filtering(subway_weighted_df1)


bike_new_total_df.stack()
bus_weighted_df.stack()
bus_weighted_df1.stack()
subway_weighted_df.stack()
subway_weighted_df1.stack()


#%%
transport_df = pd.concat([bike_new_total_df.stack(),
           bus_weighted_df.stack(),
           bus_weighted_df1.stack(),
           subway_weighted_df.stack(),
           subway_weighted_df1.stack()], axis=1)

transport_df.columns = ['bike_return', 'bus_board', 'bus_resembark', 'subway_board', 'subway_resembark']


transport_df.to_csv(r'sungsoo_prep_transport_by_dohyeon_20241115.csv')


















