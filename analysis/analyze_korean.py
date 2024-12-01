# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:22:19 2024

@author: dohyeon
"""

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import shap
from sungsoo_preprocess_final import *
import seaborn as sns




font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

def split_train_test(total_df):

    scaler = StandardScaler()
    
    train_df = total_df.loc['2023']
    test_df = total_df.loc['2024']
    
    train_x = train_df[train_df.columns[:-2]].values
    train_y = train_df.y.values
    
    test_x = test_df[test_df.columns[:-2]].values
    test_y = test_df.y.values
    
    
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    
    train_x = np.concatenate([train_x, train_df.paeup.values.reshape(-1,1)], axis=1)
    test_x = np.concatenate([test_x, test_df.paeup.values.reshape(-1,1)], axis=1)

    return train_df, test_df, train_x, train_y, test_x, test_y


def train(train_x, train_y):

    xgb_model = XGBRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0,
        colsample_bytree=0.8
    )

    xgb_model.fit(train_x, train_y) 

    return xgb_model

def visualize_importance(fitted_model, test_df):

    # 변수 중요도 추출
    importances = fitted_model.feature_importances_
    feature_names = test_df.columns[:-1]
    
    # 중요도 데이터프레임 생성
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # 변수 중요도 시각화
    plt.figure(figsize=(12, 10))
    plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('Feature_Importance.png')
    #plt.show()
    plt.close()

def visualize_grid_paeup_distribution(train_test_df1, target_variable, target_scope):
    train_test_df1= train_test_df1.reset_index().rename(columns = {'level_0' : 'date', 'level_1' : 'polygon_id1'} ).sort_index()

    #train_test_df1 = train_test_df1.loc[train_test_df1['polygon_id1'] != '다사60ba49ab']

    print(train_test_df1.columns)

    grid_name_mapping = {
        '다사60aa49bb': '뚝섬역',   # 뚝섬역
        '다사60ba49ab': '가죽거리',   # 가죽거리
        '다사60ba49bb': '정비소',    # 정비소
        '다사60ba48bb': '뚝도시장', # 뚝도시장

    }

    #train_test_df1 = train_test_df1.loc[train_test_df1['polygon_id1'] == '다사60ba48bb']

    paeup_name_mapping = {
        0: '지속',   
        1: '폐업',   
    }

    train_test_df1['polygon_id1'] = train_test_df1['polygon_id1'].replace(grid_name_mapping)
    train_test_df1['Closed'] = train_test_df1['Closed'].replace(paeup_name_mapping)

    grid_color_mapping = {
        '뚝섬역': 'SkyBlue',   # 뚝섬역
        '가죽거리': 'MediumBlue',   # 가죽거리
        '정비소': 'LightCoral',    # 성결교회
        '뚝도시장': 'DarkRed', # 뚝도시장
    }


    paeup_color_mapping = {
        '지속': 'blue',   #  지속
        '폐업': 'red',   # 폐업
    }


    plt.figure(figsize=(12, 6))  # Adjust these values as needed

    # Use the color mapping in the plot

    if target_scope == 'grid':
        sns.lineplot(data=train_test_df1, x='date', y=target_variable, hue='polygon_id1', palette=grid_color_mapping)
        target_suffix = 'grid'
    elif target_scope == 'paeup':
        sns.lineplot(data=train_test_df1, x='date', y=target_variable, hue='Closed', palette = paeup_color_mapping, ci=None)
        target_suffix = 'paeup'

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel(f'{target_variable}')
    plt.title(f'{target_variable} by Date and Polygon ID')

    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig(f'images/influx_{target_suffix}_{target_variable}.png')
    plt.show()
    plt.close()

def visualize_within_grid(train_test_df1, target_grid, ratio_flag = False):
    train_test_df1= train_test_df1.reset_index().rename(columns = {'level_0' : 'date', 'level_1' : 'polygon_id1'} ).sort_index()

    grid_name_mapping = {
        '다사60aa49bb': '뚝섬역',   # 뚝섬역
        '다사60ba49ab': '가죽거리',   # 가죽거리
        '다사60ba49bb': '정비소',    # 정비소
        '다사60ba48bb': '뚝도시장', # 뚝도시장

    }

    train_test_df1 = train_test_df1.loc[train_test_df1["polygon_id1"] == target_grid]

    train_test_df1['polygon_id1'] = train_test_df1['polygon_id1'].replace(grid_name_mapping)

    sum_male_columns = [column for column in train_test_df1.columns if column.startswith("sum_male") and column.endswith("1")]

    sum_feml_columns = [column for column in train_test_df1.columns if column.startswith("sum_feml") and column.endswith("1")]

    train_test_male_df1 = train_test_df1[['date', 'polygon_id1'] + sum_male_columns]
    train_test_feml_df1 = train_test_df1[['date', 'polygon_id1'] + sum_feml_columns]

    print(train_test_male_df1)

    long_male_df = train_test_male_df1.melt(
        id_vars=['date', 'polygon_id1'],  # Columns to keep
        value_vars=sum_male_columns,      # Columns to unpivot
        var_name='male_type',             # New column name for variable
        value_name='value'                # New column name for values
    )

    

    long_feml_df = train_test_feml_df1.melt(
        id_vars=['date', 'polygon_id1'],  # Columns to keep
        value_vars=sum_feml_columns,      # Columns to unpivot
        var_name='feml_type',             # New column name for variable
        value_name='value'                # New column name for values
    )

    # Create the plot
    plt.figure(figsize=(12, 6))

    male_palette = sns.color_palette("hls", n_colors=len(long_male_df['male_type'].unique()))
    feml_palette = sns.color_palette("husl", n_colors=len(long_feml_df['feml_type'].unique()))

    avg_male_values = long_male_df.groupby('male_type')['value'].mean().reset_index()
    avg_feml_values = long_feml_df.groupby('feml_type')['value'].mean().reset_index()
    # Step 2: Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=avg_male_values,
        x='male_type',
        y='value',
        palette=male_palette  # Use the same palette for consistency
    )

    # Customize the plot
    plt.title('Average Values for Each Male Type')
    plt.xlabel('Male Type')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'images/influx_ratio_male_{target_grid}.png')

    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=avg_feml_values,
        x='feml_type',
        y='value',
        palette=feml_palette  # Use the same palette for consistency
    )

    # Customize the plot
    plt.title('Average Values for Each Female Type')
    plt.xlabel('Female Type')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'images/influx_ratio_feml_{target_grid}.png')

    plt.close()

if __name__=="__main__":
    keep_grid = ['다사60aa49bb' # 뚝섬역
                 , '다사60ba49ab' # 가죽거리
                 ]
    
    paeup_grid = ['다사60ba49bb' # 성결교회
                  , '다사60ba48bb' # 뚝도시장
                  ]

    
    """
    cate_mask = 0

    raw_df = pd.read_csv("card_category_conditioned_processed.csv", encoding='cp949', index_col=0)
    raw_df = raw_df[~pd.isna(raw_df['age'])]
    raw_df['date'] = pd.DatetimeIndex(raw_df['date'])
    raw_df.set_index('date', inplace=True)

    raw_df.loc[:, '폐업'] = '중립'
    raw_df.loc[raw_df.code250.isin(paeup_grid), '폐업'] = '폐업'
    raw_df.loc[raw_df.code250.isin(keep_grid), '폐업'] = '지속'

    #한식은 0, 외국음식은 1
    korean_raw = raw_df[raw_df['cate_mask'] == cate_mask]

    korean_raw = korean_raw.loc[korean_raw['code250'].isin(paeup_grid + keep_grid)]

    korean_raw = korean_raw.groupby(['date', 'code250']).sum()['sum_use_count']

    korean_raw = korean_raw.reset_index()

    print(korean_raw)
    """

    

    train_test_df1 = main(0, keep_grid= keep_grid, paeup_grid = paeup_grid)
    
    #sum_male_columns = [column for column in train_test_df1.columns if column.startswith("sum_male") and column.endswith("1")]
    #sum_feml_columns = [column for column in train_test_df1.columns if column.startswith("sum_feml") and column.endswith("1")]
    target_variable = 'y'
    target_scope = 'grid'
    visualize_grid_paeup_distribution(train_test_df1, target_variable, target_scope)
    #for target_variable in sum_male_columns:
    #    visualize_grid_paeup_distribution(train_test_df1, target_variable, target_scope)

    #for target_variable in sum_feml_columns:
    #    visualize_grid_paeup_distribution(train_test_df1, target_variable, target_scope)

    target_grid = '다사60ba49bb'
    #visualize_within_grid(train_test_df1, target_grid, ratio_flag = True)

    """
    _, test_df, train_x, train_y, test_x, test_y = split_train_test(train_test_df1)

    fitted_model = train(train_x, train_y)

    y_pred = fitted_model.predict(test_x)
    
    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    visualize_importance(fitted_model, test_df)


    explainer = shap.TreeExplainer(fitted_model)
    
    # 테스트 데이터에 대해 SHAP 값 계산
    #%%
    shap_values = explainer.shap_values(test_x)
    plt.figure()
    shap.summary_plot(shap_values, test_df[test_df.columns[:-1]], show=False)
    plt.savefig("shap_test_df_summary_plot.png")
    plt.close()
    shap_df = pd.DataFrame(data=shap_values, columns=test_df.drop(columns=["y"]).columns, index=test_df.index)
    

    ##################################################### 탁 추가
    group_0 = test_df[test_df["paeup"] == 0]  # 지속 격자
    group_1 = test_df[test_df["paeup"] == 1]  # 폐업 격자

    print("test_x shape:", test_x.shape)
    condition = test_x[:, -1] == 0
    print("Condition shape:", condition.shape)
    print("Condition values:", condition)

    filtered_test_x = test_x[condition, :]
    print("Filtered test_x shape:", filtered_test_x.shape)

    shap_values_0 = explainer.shap_values(test_x[test_x[:,-1]==0,:])
    plt.figure()
    shap.summary_plot(shap_values_0, group_0.drop(columns=["y"]), show=False)
    plt.savefig("shap_keep_df_summary_plot.png")
    plt.close()

    plt.figure()
    shap_values_1 = explainer.shap_values(test_x[test_x[:,-1]==1,:])
    shap.summary_plot(shap_values_1, group_1.drop(columns=["y"]), show=False)
    plt.savefig("shap_paeup_df_summary_plot.png")
    plt.close()

    
    #%% cal shap_diff


    closure_shap_mean = shap_df[test_df['paeup'] == 1].mean()
    continuous_shap_mean = shap_df[test_df['paeup'] == 0].mean()
    
    # 평균 차이 계산
    shap_diff = closure_shap_mean - continuous_shap_mean
    print(shap_diff.sort_values(ascending=False))
    
    #%%

    plt.figure()
    target_variable = 'num'

    shap_df[target_variable].unstack().plot()
    plt.savefig(f'shap_difference_gridwise_{target_variable}.png')
    plt.close()

    new_train_test_df1 = train_test_df1.reset_index()

    new_train_test_df1 = new_train_test_df1.rename(columns = { 'level_0' : 'date' , 'level_1' : 'polygon_id1'})

    #print(new_train_test_df1)

    #print(new_train_test_df1.columns)

    new_train_test_df1 = new_train_test_df1.loc[new_train_test_df1['polygon_id1'].isin(['다사60ba48bb', '다사60ba49ab'])]

    group_df = new_train_test_df1.groupby(['polygon_id1']).mean()[[
                                                                'per_deposit',
                                                                #'bus_board', 
                                                                #'bus_resembark', 
                                                                #'subway_board',
                                                                #'subway_resembark',
                                                                #'bike_return',
                                                                #'sum_total_cnt',
                                                                'sum_male_60_69', 
                                                                'sum_male_60_691', 
                                                                #'sum_feml_50_59',
                                                                #'sum_feml_20_29',
                                                                #'num',
                                                                'paeup', 'y']]

    print(group_df)
    """
