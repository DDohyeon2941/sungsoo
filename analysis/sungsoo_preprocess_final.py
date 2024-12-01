# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:00:31 2024

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


def generate_paeup_subgroup_colors(paeup_color_mapping, num_subgroups):
    subgroup_color_mapping = {}
    
    for paeup, base_color in paeup_color_mapping.items():
        # Convert the base color name to a colormap
        colormap = cm.get_cmap(f"{base_color.capitalize()}s", num_subgroups)
        # Generate shades for subgroups
        shades = [colormap(i) for i in range(colormap.N)]
        # Map shades to subgroup IDs
        subgroup_color_mapping[paeup] = {subgroup_id: shades[subgroup_id] for subgroup_id in range(num_subgroups)}
    
    return subgroup_color_mapping

def get_saup(cate_mask):
    saup_df = pd.read_csv(r'saup_number.csv', index_col=0, header=[0,1])
    var5 = pd.DataFrame(index=saup_df.index, columns=[2023,2024])
    
    var5.loc[:,2023] = saup_df[str(cate_mask)].fillna(0)[['2024.0','2025.0']].sum(axis=1)
    var5.loc[:,2024] = saup_df[str(cate_mask)].fillna(0)['2025.0']
    
    var5 = var5.stack().reset_index()
    var5.columns = ['grid1','train_year','num']
    return var5

def get_deposit(cate_mask):

    new_temp_df = pd.read_csv(r'sungsoo_deposit_prep_dataset.csv')

    new_temp_df = new_temp_df.loc[(new_temp_df['환산_보증금']>=6.00e+06)&((new_temp_df['환산_보증금']<= 8.50e+08))].reset_index(drop=True)
    new_temp_df.loc[:, 'per_deposit'] =  new_temp_df['환산_보증금'] / new_temp_df['면적']
    new_temp_df.loc[:,'start_year'] = [int(str(xx)[:4]) for xx in new_temp_df['개업_일자']]

    meals_cate = [
        "일식", "일식 음식점업", "일식(라멘)", "일식(우동)", "일식 음식점업(이자카야)", "일본음식점업", 
        "라멘", "우동,김밥,국수", "퓨전 일식", "퓨전일식", "이자카야", 
        "중식", "중식 음식점업", "중식 요리 전문점", "중화음식점", "중식\u3000음식점업", "퓨전중식", 
        "양식", "양식 음식점업", "서양식 음식점업", "서양음식", "서양식 일반 음식점", "서양음식점", 
        "서양식", "서양식, 퓨전음식", "피자(양식)", "멕시칸요리", "이탈리아음식", "프렌치음식", 
        "경양식", "햄버거전문점", "스테이크, 햄버거", "퓨전양식"
    ]

    korean_cate = [
        "한식", "한식 음식점업", "한식 일반 음식점업", "한식 음식점업", "한식(육류)", "한식점", "한식점업", 
        "한식업", "한식,설렁탕", "한식, 분식", "한식 전문점", "한식 해산물 요리 전문점", "한식 닭요리", 
        "한식(감자탕)", "한식(일반음식점)", "한식 면 요리 전문점", "한식, 아시안요리 음식점, 케이터링(출장음식)", 
        "퓨전 한식", "퓨전한식", "한식 퓨전 음식", "한식(양꼬치)", "한식 육류 요리 전문점", 
        "한식\u3000음식점업", "한식 백반", "백반", "전, 한식", "꼼장어", "감자탕", "부대찌개", "족발전문점"
    ]

    new_temp_df.loc[:, 'meal_mask'] = [1 if xx in meals_cate else 0 for xx in new_temp_df['세부_업종_이름']]
    new_temp_df.loc[:, 'korean_mask'] = [1 if xx in korean_cate else 0 for xx in new_temp_df['세부_업종_이름']]
    new_temp_df.loc[:, 'train_year'] = [2023 if xx <=2023 else 2024 for xx in new_temp_df['start_year']]

    ####

    #한식이면 korean_mask == 1, 외국음식이면 meal_mask == 1로
    if cate_mask == 0:
        new_temp_df1 = new_temp_df.loc[new_temp_df.korean_mask==1].reset_index(drop=True)
    elif cate_mask == 1:
        new_temp_df1 = new_temp_df.loc[new_temp_df.meal_mask==1].reset_index(drop=True)
    elif cate_mask == 2:
        new_temp_df1 = new_temp_df.loc[new_temp_df.cafe_mask==1].reset_index(drop=True)


    var3 = (
        new_temp_df1.groupby(["0", "train_year"])
        .mean()[["면적", "환산_보증금","per_deposit"]]
        .reset_index()
    )
    pivot_df = var3.pivot(index="0", columns="train_year", values=["면적", "환산_보증금","per_deposit"])
    pivot_df = pivot_df.fillna(method="ffill", axis=1)

    var3 = pivot_df.stack("train_year").reset_index()
    var3.columns = ["0", "train_year", "면적", "환산_보증금","per_deposit"]


    return var3


def get_target(cate_mask, keep_grid, paeup_grid):
    raw_df = pd.read_csv("card_category_conditioned_processed.csv", encoding='cp949', index_col=0)
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

def get_transport(var3, var5, keep_grid, paeup_grid):
    new_transport_df = pd.read_csv(r'sungsoo_prep_transport_by_dohyeon_20241115.csv')
    
    new_transport_df.loc[:, 'use_date'] = pd.to_datetime(new_transport_df['use_date'])
    
    new_transport_df = new_transport_df.loc[new_transport_df['Unnamed: 1'].isin(np.append(paeup_grid,keep_grid))]
    
    new_transport_df.loc[:,'train_year'] = [xx.year for xx in new_transport_df['use_date']]
    
    new_transport_df = pd.merge(left=new_transport_df, right= var3, left_on = ['Unnamed: 1','train_year'], right_on=['0', 'train_year'])

    new_transport_df = pd.merge(left=new_transport_df, right= var5, left_on = ['Unnamed: 1','train_year'], right_on=['grid1', 'train_year'])

    new_transport_df.loc[:,'bus_ratio'] = new_transport_df['bus_board'] /  new_transport_df['bus_resembark']
    new_transport_df.loc[:,'subway_ratio'] = new_transport_df['subway_board'] /  new_transport_df['subway_resembark']
    
    
    new_transport_df.loc[:, 'paeup'] = [1 if xx in paeup_grid else 0 for xx in new_transport_df['Unnamed: 1']]
    new_transport_df.drop(columns=['train_year','0','grid1'], inplace=True)
    
    new_transport_df.set_index(['use_date','Unnamed: 1'], inplace=True)

    return new_transport_df

def get_moving(new_transport_df):

    moving_df_2023 = pd.read_csv(r'sungsoo_grid_arrival_2023_20241114.csv')
    moving_df_2024 = pd.read_csv(r'sungsoo_grid_arrival_2024_20241114.csv')

    moving_df_20231 = pd.read_csv(r'sungsoo_grid_arrival_2023_20241125.csv')
    moving_df_20241 = pd.read_csv(r'sungsoo_grid_arrival_2024_20241125.csv')


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


def get_weather(paeup_grid, keep_grid):
    weather_df = pd.read_csv(r'building_transport_board_weather_20241114.csv', encoding='cp949')
    weather_df = weather_df[['polygon_id1','일시', '건물높이', '건축면적',
    '건폐율', '공지지가','연면적','용적율', 'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균습도(%rh)', '최저습도(%rh)']]

    new_weather_df = weather_df.loc[weather_df.polygon_id1.isin(np.append(paeup_grid,keep_grid))]
    new_weather_df.loc[:,'일시'] = pd.DatetimeIndex(new_weather_df['일시'])
    new_weather_df.set_index(['일시','polygon_id1'], inplace=True)

    return new_weather_df


def main(cate_mask, keep_grid, paeup_grid):

    var5 = get_saup(cate_mask)
    var3 = get_deposit(cate_mask)
    target0 = get_target(cate_mask, keep_grid, paeup_grid)
    new_transport_df = get_transport(var3, var5, keep_grid, paeup_grid)
    moving_df_2023_2024 = get_moving(new_transport_df)
    new_weather_df = get_weather(paeup_grid, keep_grid)

    train_test_df1 = pd.concat([moving_df_2023_2024, new_weather_df, new_transport_df, target0.loc[new_transport_df.index]], axis=1)
    train_test_df1.rename(columns={0:'y'}, inplace=True)

    
    train_test_df1.drop(columns=['건축면적', '연면적','용적율','건폐율','건물높이','sum_total_cnt','sum_total_cnt1','공지지가','환산_보증금', 'avg_move_dist1', 'avg_move_time1'], inplace=True)

    train_test_df1 = train_test_df1.rename(rename_mapper, axis = 1)

    train_test_df1= train_test_df1.reset_index().set_index(['level_0','level_1']).sort_index()

    return train_test_df1


#train_test_df1 = main(1, keep_grid= ['다사59bb49bb', '다사60ba48bb'],paeup_grid = ['다사60ba49bb'])
