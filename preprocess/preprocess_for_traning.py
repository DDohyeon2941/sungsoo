# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:55:43 2025

@author: dohyeon
"""

import pandas as pd
import numpy as np
from matplotlib import cm

pd.options.display.float_format = '{:.2f}'.format


############### rename_mapper 추가

rename_mapper = {
        "avg_move_dist": "Avg Dist",
        "avg_move_time": "Avg Time",
        "sum_male_00_19": "Count Male Under 20",
        "sum_male_20_29": "Count Male 20-29",
        "sum_male_30_39": "Count Male 30-39",
        "sum_male_40_49": "Count Male 40-49",
        "sum_male_50_59": "Count Male 50-59",
        "sum_male_60_69": "Count Male 60-69",
        "sum_male_70_79": "Count Male 70-79",
        "sum_male_80_89": "Count Male 80-89",
        "sum_feml_00_19": "Count Female Under 20",
        "sum_feml_20_29": "Count Female 20-29",
        "sum_feml_30_39": "Count Female 30-39",
        "sum_feml_40_49": "Count Female 40-49",
        "sum_feml_50_59": "Count Female 50-59",
        "sum_feml_60_69": "Count Female 60-69",
        "sum_feml_70_79": "Count Female 70-79",
        "sum_feml_80_89": "Count Female 80-89",
        "sum_male_00_191": "Influx Ratio Male Under 20",
        "sum_male_20_291": "Influx Ratio Male 20-29",
        "sum_male_30_391": "Influx Ratio Male 30-39",
        "sum_male_40_491": "Influx Ratio Male 40-49",
        "sum_male_50_591": "Influx Ratio Male 50-59",
        "sum_male_60_691": "Influx Ratio Male 60-69",
        "sum_male_70_791": "Influx Ratio Male 70-79",
        "sum_male_80_891": "Influx Ratio Male 80-89",
        "sum_feml_00_191": "Influx Ratio Female Under 20",
        "sum_feml_20_291": "Influx Ratio Female 20-29",
        "sum_feml_30_391": "Influx Ratio Female 30-39",
        "sum_feml_40_491": "Influx Ratio Female 40-49",
        "sum_feml_50_591": "Influx Ratio Female 50-59",
        "sum_feml_60_691": "Influx Ratio Female 60-69",
        "sum_feml_70_791": "Influx Ratio Female 70-79",
        "sum_feml_80_891": "Influx Ratio Female 80-89",
        "bus_board": "Count Bording Bus",
        "bus_resembark": "Count Alighting Bus",
        "subway_board": "Count Bording Subway",
        "subway_resembark": "Count Alighting Subway",
        "bike_return": "Count Return Bike",
        "bus_ratio": "Ratio Bording Bus",
        "subway_ratio": "Ratio Bording Subway",
        "SO2": "SO2",
        "CO": "CO",
        "O3": "O3",
        "NO2": "NO2",
        "PM10": "PM10",
        "PM25": "Pm25",
        "평균기온(℃)": "Avg Temporature",
        "최고기온(℃)": "Max Temporature",
        "최저기온(℃)": "Min Temporature",
        "평균습도(%rh)": "Avg Humanity",
        "최저습도(%rh)": "Min Humanity",
        "면적": "Area",
        "num": "Store",
        "per_deposit": "Deposit",
        "paeup": "Closed"
    }

############### 시각화 color 추가

paeup_color_mapping = {
        0: 'blue',   
        1: 'red'
    }   

grid_color_mapping = {
    '지속1': 'SkyBlue',
    '지속2': 'MediumBlue',
    
    '폐업1': 'LightCoral',
    '폐업2': 'DarkRed'
}

def get_saup(cate_mask, saup_num_df):
    #saup_df = pd.read_csv(r'saup_number.csv', index_col=0, header=[0,1])
    var5 = pd.DataFrame(index=saup_num_df.index, columns=[2023,2024])
    
    var5.loc[:,2023] = saup_num_df[str(cate_mask)].fillna(0)[['2024.0','2025.0']].sum(axis=1)
    var5.loc[:,2024] = saup_num_df[str(cate_mask)].fillna(0)['2025.0']
    
    var5 = var5.stack().reset_index()
    var5.columns = ['grid1','train_year','num']
    return var5


def index_transport_for_grid(var5, new_transport_df, keep_grid, paeup_grid):
    #new_transport_df = pd.read_csv(r'sungsoo_prep_transport_by_dohyeon_20241115.csv')
    
    new_transport_df.loc[:, 'use_date'] = pd.to_datetime(new_transport_df['use_date'])
    
    new_transport_df = new_transport_df.loc[new_transport_df['Unnamed: 1'].isin(np.append(paeup_grid,keep_grid))]
    
    new_transport_df.loc[:,'train_year'] = [xx.year for xx in new_transport_df['use_date']]
    
    #new_transport_df = pd.merge(left=new_transport_df, right= var3, left_on = ['Unnamed: 1','train_year'], right_on=['0', 'train_year'])

    new_transport_df = pd.merge(left=new_transport_df, right= var5, left_on = ['Unnamed: 1','train_year'], right_on=['grid1', 'train_year'])

    new_transport_df.loc[:,'bus_ratio'] = new_transport_df['bus_board'] /  new_transport_df['bus_resembark']
    new_transport_df.loc[:,'subway_ratio'] = new_transport_df['subway_board'] /  new_transport_df['subway_resembark']
    
    
    new_transport_df.loc[:, 'paeup'] = [1 if xx in paeup_grid else 0 for xx in new_transport_df['Unnamed: 1']]
    new_transport_df.drop(columns=['train_year','grid1'], inplace=True)
    
    new_transport_df.set_index(['use_date','Unnamed: 1'], inplace=True)

    return new_transport_df

def get_moving(new_transport_df):

    moving_df_2023 = pd.read_csv(r'moving\sungsoo_grid_arrival_2023_20241114.csv')
    moving_df_2024 = pd.read_csv(r'moving\sungsoo_grid_arrival_2024_20241114.csv')

    moving_df_20231 = pd.read_csv(r'moving\sungsoo_grid_arrival_2023_20241125.csv')
    moving_df_20241 = pd.read_csv(r'moving\sungsoo_grid_arrival_2024_20241125.csv')


    moving_df_2023.loc[:, 'date'] = pd.to_datetime(moving_df_2023['date'])
    moving_df_2023.set_index(['date','d_cell_id'], inplace=True)

    moving_df_20231.loc[:, 'date'] = pd.to_datetime(moving_df_20231['date'])
    moving_df_20231.set_index(['date','d_cell_id'], inplace=True)


    moving_df_2024.loc[:, 'date'] = pd.to_datetime(moving_df_2024['date'])
    moving_df_2024.set_index(['date','d_cell_id'], inplace=True)


    moving_df_20241.loc[:, 'date'] = pd.to_datetime(moving_df_20241['date'])
    moving_df_20241.set_index(['date','d_cell_id'], inplace=True)

    moving_df_2023_main = pd.concat([moving_df_2023.loc[new_transport_df.loc['2023'].index],    moving_df_20231.loc[new_transport_df.loc['2023'].index]], axis=1)

    moving_df_2024_main = pd.concat([moving_df_2024.loc[new_transport_df.loc['2024'].index],    moving_df_20241.loc[new_transport_df.loc['2024'].index]], axis=1)

    moving_df_2023_2024 = pd.concat([moving_df_2023_main, moving_df_2024_main])

    return moving_df_2023_2024


def get_target(cate_mask, raw_df, keep_grid, paeup_grid):
    #raw_df = pd.read_csv("card_category_conditioned_processed.csv", encoding='cp949', index_col=0)
    raw_df = raw_df[~pd.isna(raw_df['age'])]
    raw_df['date'] = pd.DatetimeIndex(raw_df['date'])
    raw_df.set_index('date', inplace=True)

    raw_df.loc[:, '폐업'] = '중립'
    raw_df.loc[raw_df.code250.isin(paeup_grid), '폐업'] = '폐업'
    raw_df.loc[raw_df.code250.isin(keep_grid), '폐업'] = '지속'

    #한식은 0, 외국음식은 1
    cafe_raw = raw_df[raw_df['cate_mask'] == cate_mask]

    target0  = np.log(cafe_raw[cafe_raw['폐업'] != '중립'].groupby(['date','code250']).sum()['sum_amount'].unstack(fill_value=0).stack()+1)

    return target0


def get_weather(weather_df):
    #weather_df = pd.read_csv(r'building_transport_board_weather_20241114.csv', encoding='cp949')
    weather_df = weather_df[['일시', 'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균습도(%rh)', '최저습도(%rh)']]
    weather_df['일시'] = pd.to_datetime(weather_df['일시'])
    return weather_df


def conv_weather(weather_df, transport_df, keep_grid, paeup_grid):
    filtered_df = transport_df.loc[transport_df['Unnamed: 1'].isin(np.append(keep_grid,paeup_grid))]
    filtered_df['use_date'] = pd.to_datetime(filtered_df['use_date'])
    return pd.merge(filtered_df, weather_df, left_on='use_date', right_on = '일시',how='left')

def main(cate_mask, paeup_grid, keep_grid):
    saup_df = pd.read_csv(r'paeup\saup_number.csv', index_col=0, header=[0,1])
    sales_df = pd.read_csv(r'sales\card_category_conditioned_processed.csv', encoding='cp949', index_col=0)
    weather_df = get_weather(pd.read_csv(r'weather\weather.csv').dropna())

    var5 = get_saup(cate_mask, saup_df)

    transport_df = pd.read_csv(r'transport\sungsoo_prep_transport_by_dohyeon_20241115.csv')

    new_transport_df = index_transport_for_grid(var5, transport_df, keep_grid, paeup_grid)
    new_moving_df = get_moving(new_transport_df)


    new_target_df = get_target(cate_mask, sales_df, keep_grid, paeup_grid )

    new_weather_df = conv_weather(weather_df, transport_df, keep_grid, paeup_grid)
    new_weather_df.set_index(['use_date','Unnamed: 1'], inplace=True)


    train_test_df = pd.concat([new_weather_df, new_moving_df, new_transport_df, new_target_df], axis=1)
    train_test_df.rename(columns={0:'y'}, inplace=True)
    train_test_df.drop(columns=['sum_total_cnt','sum_total_cnt1', 'avg_move_dist1', 'avg_move_time1'], inplace=True)
    train_test_df = (train_test_df.rename(rename_mapper, axis = 1)).dropna()
    train_test_df = train_test_df.reset_index().set_index(['level_0','level_1']).sort_index()
    return train_test_df


#%%
if __name__ == "__main__":

    t1 = main(0, keep_grid= ['다사59bb49ba'], paeup_grid = ['다사60ab50aa', '다사60ab49ab'])

    t1.loc['2023']
