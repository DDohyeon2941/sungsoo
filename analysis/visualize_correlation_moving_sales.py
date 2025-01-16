# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:22:20 2025

@author: dohyeon
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

"""target year를 2023, 2024로 해서 각각 산출"""

target_year = 2023

if __name__ == "__main__":
    moving_df = pd.read_csv(r'sungsoo_grid_arrival_%s.csv'%(target_year))
    polygon_df = pd.read_csv(r'sungsoo_2nd_drop_polygons_45.csv', index_col=0)
    grid_dic = dict(zip(polygon_df['1'],polygon_df['0']))

    moving_df = moving_df.loc[moving_df['d_cell_id'].isin(polygon_df['1'])].reset_index(drop=True)

    moving_df.loc[:, 'd_cell_id'] = [grid_dic[xx] for xx in moving_df['d_cell_id']]
    

    #%% 1,2차 필터링된 매출액 데이터
    
    whole_df = pd.read_csv(r'whole_data_45_grid_51_cate.csv')
    
    whole_df['date'] = pd.to_datetime(whole_df['date'])
    whole_df = (whole_df.loc[whole_df['date'].dt.year == target_year]).reset_index(drop=True)
    
    
    #%% 3차 필터링 후, 요식 대분류에 대해서만 추출한 매출액 데이터

    food_sales_df = pd.read_csv(r'card_category_conditioned.csv')
    
    food_sales_df['date'] = pd.to_datetime(food_sales_df['date'])
    food_sales_df = (food_sales_df.loc[food_sales_df['date'].dt.year == target_year]).reset_index(drop=True)
    food_sales_df.loc[:,'cate_mask']= 5
    
    food_sales_df.loc[food_sales_df['category']=='한식', 'cate_mask'] = 0
    food_sales_df.loc[food_sales_df['category'].isin(['중식','양식','일식']), 'cate_mask'] = 0
    food_sales_df.loc[food_sales_df['category'].isin(['패스트푸드','커피전문점','제과점']), 'cate_mask'] = 1
    
    
    food_sales_df = food_sales_df.loc[food_sales_df.cate_mask!=5].reset_index(drop=True)
    food_sales_df = (food_sales_df.loc[food_sales_df['code250'].isin(polygon_df['0'].unique())]).reset_index(drop=True)


    #%% 일별 매출액 및 유동인구 집계량 산출 및 최댓값에 대한 비중으로 변환

    daily_sales_whole_df = whole_df[['sum_use_count','sum_amount','date']].groupby('date').mean()[['sum_use_count','sum_amount']]
    daily_sales_food_df = food_sales_df[['sum_use_count','sum_amount','date']].groupby('date').mean()[['sum_use_count','sum_amount']]
    
    daily_moving_whole_df = moving_df[['sum_total_cnt','month','day']].groupby(['month','day']).mean()['sum_total_cnt']
    
    ratio_sales_whole_df = daily_sales_whole_df['sum_amount'] / daily_sales_whole_df['sum_amount'].max()
    ratio_sales_food_df = daily_sales_food_df['sum_amount'] / daily_sales_food_df['sum_amount'].max()
    ratio_moving_whole_df = daily_moving_whole_df / daily_moving_whole_df.max()
    
    
    
    #%% 시각화
    
    fig1, axes1 = plt.subplots(2,1, figsize=(8,8))
    
    axes1[0].scatter(ratio_sales_food_df.values, ratio_moving_whole_df.values)
    axes1[0].set_xlabel("Meal Sales Ratio")
    axes1[0].set_ylabel("Moving Ratio")
    
    
    axes1[1].scatter(ratio_sales_whole_df.values, ratio_moving_whole_df.values)
    axes1[1].set_xlabel("Whole Sales Ratio")
    axes1[1].set_ylabel("Moving Ratio")
    
    axes1[0].set_title("Distribution of sales and moving ratio for each day")
    
    
    
