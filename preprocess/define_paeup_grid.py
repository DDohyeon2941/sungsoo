# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:32:26 2025

@author: dohyeon
"""

import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)



if __name__ == "__main__":
    temp_df = pd.read_csv(r'sungsoo_paeup_grid_coef_20250115.csv')
    
    t1 = temp_df.loc[temp_df['end_year']>2022].groupby(['grid1','cate_mask']).sum()['paeup'].unstack(fill_value=0)[[0,1,2]]
    t2 = temp_df.groupby(['grid1','cate_mask']).count()['paeup'].unstack(fill_value=0)[[0,1,2]]
    
    
    t3 = temp_df.loc[temp_df['end_year']<=2022].groupby(['grid1','cate_mask']).sum()['paeup'].unstack(fill_value=0)[[0,1,2]]
    t4 = t2-t3
    
    t4 = t4.loc[(t4.sum(axis=1).loc[t4.sum(axis=1)>25]).index]

    t1 = t1.loc[t4.index]
    
    active_df = t4-t1
    #%%

    ratio_df = (t1/t4).fillna(0)
    
    ratio_df = ratio_df.loc[t4.index]

    temp_df.loc[:,'start_year'] = pd.to_datetime(temp_df.start_date).dt.year
    t5=(temp_df.loc[temp_df.start_year>=2023]).groupby(['grid1','cate_mask']).count()['start_year'].unstack(fill_value=0)[[0,1,2]]

    t5 = t5.loc[active_df.index]
    
    t6 = (t5 / active_df).fillna(0)

    #%%
    """폐업률과 창업률을 기반으로 폐업격자 1차 정의

    밑에 col_val이 0이면 한식, 1이면 외국음식, 2면 카페

    """
    #%%
    col_val=0

    fig1, axes1= plt.subplots(1,1)
    axes1.scatter(ratio_df[col_val], t6[col_val])
    axes1.hlines(xmin=0, xmax=1.0,y=np.mean(t6[col_val]), color='k')
    axes1.vlines(ymin=0, ymax=1.0,x=np.mean(ratio_df[col_val]), color='k')
    axes1.set_xlabel('Closing Ratio')
    axes1.set_ylabel('Open Ratio')
    axes1.set_title('Foreign')

    #%%
    finish_more_grid = (ratio_df[col_val].loc[ratio_df[col_val]>np.mean(ratio_df[col_val])]).index.values
    start_less_grid = (t6[col_val].loc[t6[col_val]<=np.mean(t6[col_val])]).index.values
    paeup_grid = np.intersect1d(finish_more_grid, start_less_grid)

    finish_less_grid = (ratio_df[col_val].loc[ratio_df[col_val]<=np.mean(ratio_df[col_val])]).index.values
    start_more_grid = (t6[col_val].loc[t6[col_val]>np.mean(t6[col_val])]).index.values
    keep_grid = np.intersect1d(finish_less_grid, start_more_grid)

    #%%

    polygon_df =  pd.read_csv(r'sungsoo_2nd_drop_polygons_45.csv',index_col=0)

    #%%

    selected_area_df = polygon_df.loc[polygon_df['0'].isin(np.append(paeup_grid,keep_grid))].reset_index(drop=True)
    selected_area_df.loc[:,'paeup'] = [1 if xx in paeup_grid else 0 for xx in selected_area_df['0']]
    if col_val == 0:
        selected_area_df.to_csv(r'korean_paeup_masking_polygon_20250115.csv')
    elif col_val == 1:
        selected_area_df.to_csv(r'meals_paeup_masking_polygon_20250115.csv')
    elif col_val == 2:
        selected_area_df.to_csv(r'cafe_paeup_masking_polygon_20250115.csv')

    #%%
    """매출액과 매출건수를 기반해 폐업격자 2차 정의"""
    #%%
    sales_df = pd.read_csv(r'card_category_conditioned.csv')

    sales_df['date'] = pd.to_datetime(sales_df['date'])
    sales_df = (sales_df.loc[sales_df['date'].dt.year == 2023]).reset_index(drop=True)
    sales_df.loc[:,'cate_mask']= 5

    sales_df.loc[sales_df['category']=='한식', 'cate_mask'] = 0
    sales_df.loc[sales_df['category'].isin(['중식','양식','일식']), 'cate_mask'] = 1
    sales_df.loc[sales_df['category'].isin(['패스트푸드','커피전문점','제과점']), 'cate_mask'] = 2


    new_sales_df = sales_df.loc[sales_df.cate_mask!=5].reset_index(drop=True)

    new_sales_df = (new_sales_df.loc[new_sales_df['code250'].isin(polygon_df['0'].unique())]).reset_index(drop=True)

    new_sales_df.loc[:, 'sales_day'] = pd.to_datetime(new_sales_df['date']).dt.day

    sel_new_sales_df = new_sales_df.loc[(new_sales_df.cate_mask == col_val)&(new_sales_df.code250.isin(polygon_df['0'])) ].reset_index(drop=True)

    #%% 평균 매출액
    """폐업률과 창업률을 기반해 구분한 격자에 대해서, 매출액과 매출건수 시각화"""
    if col_val == 0:
        cate_str = '한식'
    elif col_val == 1:
        cate_str = '외국음식'
    elif col_val == 2:
        cate_str = '카페'

    fig1, axes1 = plt.subplots(1,1)
    
    
    (sel_new_sales_df.loc[sel_new_sales_df['code250'].isin(paeup_grid)])[['date','sum_amount']].groupby(['date']).mean()['sum_amount'].plot(ax=axes1, c='b', label='paeup')
    (sel_new_sales_df.loc[sel_new_sales_df.code250.isin(keep_grid)])[['date','sum_amount']].groupby(['date']).mean()['sum_amount'].plot(ax=axes1, c='r', label='keep')
    axes1.set_title(cate_str)
    axes1.set_ylabel("매출액")
    axes1.set_xlabel("날짜")

    axes1.legend()
    

    #%% 평균 매출 건수
    fig1, axes1 = plt.subplots(1,1)
    
    
    (sel_new_sales_df.loc[sel_new_sales_df['code250'].isin(paeup_grid)])[['date','sum_use_count']].groupby(['date']).mean()['sum_use_count'].plot(ax=axes1, c='b', label='paeup')
    (sel_new_sales_df.loc[sel_new_sales_df.code250.isin(keep_grid)])[['date','sum_use_count']].groupby(['date']).mean()['sum_use_count'].plot(ax=axes1, c='r', label='keep')
    axes1.set_title(cate_str)
    axes1.set_ylabel("매출건수")
    axes1.set_xlabel("날짜")

    axes1.legend()

    #%% 격자별 평균 매출액


    fig1, axes1 = plt.subplots(1,2, sharey=True)
    (sel_new_sales_df.loc[sel_new_sales_df.code250.isin(paeup_grid)])[['date','code250','sum_amount']].groupby(['date','code250']).mean()['sum_amount'].unstack().mean(axis=0).plot(ax=axes1[0], kind='bar')
    (sel_new_sales_df.loc[sel_new_sales_df.code250.isin(keep_grid)])[['date','code250','sum_amount']].groupby(['date','code250']).mean()['sum_amount'].unstack().mean(axis=0).plot(ax=axes1[1], kind='bar')
    
    
    axes1[0].set_title(cate_str)
    axes1[1].set_title(cate_str)
    
    
    axes1[0].set_ylabel("매출액")
    axes1[1].set_ylabel("매출액")
    
    axes1[0].set_xlabel("폐업격자")
    axes1[1].set_xlabel("지속격자")


    #%% 격자별 평균 매출 건수

    fig1, axes1 = plt.subplots(1,2, sharey=True)
    (sel_new_sales_df.loc[sel_new_sales_df.code250.isin(paeup_grid)])[['date','code250','sum_use_count']].groupby(['date','code250']).mean()['sum_use_count'].unstack().mean(axis=0).plot(ax=axes1[0], kind='bar')
    (sel_new_sales_df.loc[sel_new_sales_df.code250.isin(keep_grid)])[['date','code250','sum_use_count']].groupby(['date','code250']).mean()['sum_use_count'].unstack().mean(axis=0).plot(ax=axes1[1], kind='bar')
    
    
    axes1[0].set_title(cate_str)
    axes1[1].set_title(cate_str)
    
    
    axes1[0].set_ylabel("매출건수")
    axes1[1].set_ylabel("매출건수")
    
    axes1[0].set_xlabel("폐업격자")
    axes1[1].set_xlabel("지속격자")


    #%%
    """위 과정 이후로, 매출액 또는 매출건수가 안정적으로 높은 경우 (지속) 와 낮은 경우 (폐업)을 선정"""
    #%%





