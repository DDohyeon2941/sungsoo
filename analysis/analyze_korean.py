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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import shap
from sungsoo_preprocess_final import *
import seaborn as sns

grid_name_mapping = {
    '다사60aa49bb': '역세권 (지속)',   # 뚝섬역
    '다사60ba49ab': '가죽거리 (지속)',   # 가죽거리
    '다사60ba49bb': '정비소 (폐업)',    # 정비소
    '다사60ba48bb': '뚝도시장 (폐업)', # 뚝도시장

}

#train_test_df1 = train_test_df1.loc[train_test_df1['polygon_id1'] == '다사60ba48bb']

paeup_name_mapping = {
    0: '지속',   
    1: '폐업',   
}

grid_color_mapping = {
    '역세권 (지속)': 'SkyBlue',   # 뚝섬역
    '가죽거리 (지속)': 'MediumBlue',   # 가죽거리
    '정비소 (폐업)': 'LightCoral',    # 성결교회
    '뚝도시장 (폐업)': 'DarkRed', # 뚝도시장
}
grid_order = list(grid_color_mapping.keys())


paeup_color_mapping = {
    '지속': 'blue',   #  지속
    '폐업': 'red',   # 폐업
}

paeup_order = list(paeup_color_mapping.keys())


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
    
    train_x = np.concatenate([train_x, train_df.Closed.values.reshape(-1,1)], axis=1)
    test_x = np.concatenate([test_x, test_df.Closed.values.reshape(-1,1)], axis=1)

    return train_df, test_df, train_x, train_y, test_x, test_y

def get_smape(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    return np.sum(np.nan_to_num((np.abs(pred_y-real_y)) / ((real_y+pred_y)/2),0))/real_y.shape[0]

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
    #plt.figure(figsize=(12, 10))
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

    train_test_df1['polygon_id1'] = train_test_df1['polygon_id1'].replace(grid_name_mapping)
    train_test_df1['Closed'] = train_test_df1['Closed'].replace(paeup_name_mapping)

    #plt.figure(figsize=(12, 6))  # Adjust these values as needed

    # Use the color mapping in the plot

    if target_scope == 'grid':
        sns.lineplot(data=train_test_df1, x='date', y=target_variable, hue='polygon_id1', palette=grid_color_mapping)
        target_suffix = 'grid'
        legend_title = '격자 특성'
    elif target_scope == 'paeup':
        sns.lineplot(data=train_test_df1, x='date', y=target_variable, hue='Closed', palette = paeup_color_mapping, ci=None)
        target_suffix = 'paeup'
        legend_title = 'Closed'

    # Set labels and title
    plt.xlabel('Date')
    plt.ylabel(f'{target_variable}')
    #plt.title(f'{target_variable} by Date and Polygon ID')
    plt.legend(title=legend_title)
    
    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'images/{target_suffix}_{target_variable}_2.png')
    #plt.show()
    plt.close()

def visualize_grid_paeup_bar_distribution(train_test_df1, target_variable):
    train_test_df1 = train_test_df1.reset_index().rename(columns={'level_0': 'date', 'level_1': 'polygon_id1'}).sort_index()

    grid_name_mapping = {
        '다사60aa49bb': '역세권 (지속)',   # 뚝섬역
        '다사60ba49ab': '가죽거리 (지속)',   # 가죽거리
        '다사60ba49bb': '정비소 (폐업)',    # 정비소
        '다사60ba48bb': '뚝도시장 (폐업)', # 뚝도시장
    }

    # Ensure new train_test_df1 is correctly set up
    new_train_test_df1 = train_test_df1.rename(columns={"level_0": "date", "level_1": "polygon_id1"})

    # Filter by relevant categories
    new_train_test_df1 = new_train_test_df1.loc[new_train_test_df1["polygon_id1"].isin(keep_grid + paeup_grid)]

    # Replace polygon_id1 values with mapped names
    new_train_test_df1["polygon_id1"] = new_train_test_df1["polygon_id1"].replace(grid_name_mapping)

    # Filter data for each category
    filtered_data1 = new_train_test_df1[new_train_test_df1["polygon_id1"] == "역세권 (지속)"]
    filtered_data2 = new_train_test_df1[new_train_test_df1["polygon_id1"] == "가죽거리 (지속)"]
    filtered_data3 = new_train_test_df1[new_train_test_df1["polygon_id1"] == "정비소 (폐업)"]
    filtered_data4 = new_train_test_df1[new_train_test_df1["polygon_id1"] == "뚝도시장 (폐업)"]

    # Create a DataFrame with mean values for each category
    mean_data_age_list = pd.DataFrame(
        {
            "Value": [
                filtered_data1[target_variable].mean(),
                filtered_data2[target_variable].mean(),
                filtered_data3[target_variable].mean(),
                filtered_data4[target_variable].mean()
            ]
        },
        index=["역세권 (지속)", "가죽거리 (지속)", "정비소 (폐업)", "뚝도시장 (폐업)"]  # Providing the index
    )

    # Plot using seaborn to get better control over color mapping
    sns.barplot(
        data=mean_data_age_list.reset_index(),  # Convert DataFrame to have 'index' as a column for seaborn
        x='index',
        y='Value',
        palette=['SkyBlue', 'MediumBlue', 'LightCoral', 'DarkRed']  # Define custom colors for each category
    )

    plt.xticks(rotation=45)
    #plt.legend(title="격자 특성", loc="lower right")
    plt.xlabel("Category")
    plt.ylabel(f'{target_variable} Mean Value')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'images/bar_{target_variable}.png')
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

def get_dohyeon_evaluation(test_df, y_pred, test_y, train_x, train_y, paeup_grid, keep_grid, fitted_model):
    test_df1 = pd.DataFrame(index=test_df.index)
    test_df1.loc[:,'pred_y'] = y_pred

    paeup_real_y = np.exp(test_df['y'].unstack()[paeup_grid].stack().values)-1
    paeup_pred_y = np.exp(test_df1.unstack()['pred_y'][paeup_grid].stack().values)-1

    keep_real_y = np.exp(test_df['y'].unstack()[keep_grid].values)-1
    keep_pred_y = np.exp(test_df1.unstack()['pred_y'][keep_grid].values)-1

    print('rmse) whole: %f, paeup: %f, keep: %s'%(    mean_squared_error(np.exp(test_y)-1, np.exp(y_pred)-1)**0.5,
                                                                         mean_squared_error(paeup_real_y,
                                                                                            paeup_pred_y)**0.5,
                                                                         mean_squared_error(keep_real_y,
                                                                                            keep_pred_y)**0.5))

    print('mae) whole: %f, paeup: %f, keep: %s'%(mean_absolute_error(np.exp(test_y)-1, np.exp(y_pred)-1),
                                                 mean_absolute_error(paeup_real_y, paeup_pred_y),
                                                 mean_absolute_error(keep_real_y, keep_pred_y)))


    print('mape) whole: %f, paeup: %f, keep: %s'%(mean_absolute_percentage_error(np.exp(test_y[test_y>0])-1, np.exp(y_pred[test_y>0])-1),
                                       mean_absolute_percentage_error(paeup_real_y[np.where(paeup_real_y>0)[0]],
                                                                      paeup_pred_y[np.where(paeup_real_y>0)[0]]),
                                       mean_absolute_percentage_error(keep_real_y[np.where(keep_real_y>0)[0]],
                                                                      keep_pred_y[np.where(keep_real_y>0)[0]])
                                       ))

    print('smape) whole: %f, paeup: %f, keep: %s'%(get_smape(np.exp(test_y)-1, np.exp(y_pred)-1),
                                                   get_smape(paeup_real_y, paeup_pred_y),
                                                   get_smape(keep_real_y, keep_pred_y)))

    r2_test_whole = r2_score(np.exp(test_y)-1, np.exp(y_pred)-1)
    r2_train_whole = fitted_model.score(train_x,train_y)

    print(f"R2 whole Score: {r2_test_whole:.2f}")
    print(f"R2 whole Score1: {r2_train_whole:.2f}")

def get_evaluations(test_y, y_pred):
    y_pred = np.exp(y_pred) - 1
    test_y = np.exp(test_y) - 1

    mse = mean_squared_error(test_y, y_pred)
    smape = get_smape(test_y, y_pred)
    rmse = np.sqrt(mse)
    non_zero_mask = test_y != 0
    mae = mean_absolute_error(test_y, y_pred)
    mape = mean_absolute_percentage_error(test_y[non_zero_mask], y_pred[non_zero_mask])
    

    print(f"MSE: {mse:.2f}")
    print(f"RMSE Score: {rmse:.2f}")
    print(f"MAE Score: {mae:.2f}")
    print(f"MAPE Score: {mape:.2f}")
    print(f"SMAPE Score: {smape:.2f}")
    

def visualize_raw_df(cate_mask, paeup_grid, keep_grid, target_variable):

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

    korean_raw = korean_raw.groupby(['date', 'code250']).sum()[target_variable]

    korean_raw = korean_raw.reset_index()

    korean_raw['code250'] = korean_raw['code250'].replace(grid_name_mapping)

    print(korean_raw)

    sns.lineplot(data=korean_raw, x='date', y='sum_use_count', hue='code250', palette=grid_color_mapping)
    plt.xlabel('Date')
    plt.ylabel(f'{target_variable}')
    plt.legend(title = '격자 특성')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'images/{target_variable}.png')
    plt.close()
    #plt.show()

def get_shap(fitted_model, test_df, test_x):
    visualize_importance(fitted_model, test_df)

    explainer = shap.TreeExplainer(fitted_model)
    
    # 테스트 데이터에 대해 SHAP 값 계산
    #%%
    shap_values = explainer.shap_values(test_x)
    plt.figure()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    shap.summary_plot(shap_values, test_df[test_df.columns[:-1]], show=False)
    plt.tight_layout()
    plt.savefig("images/shap_test_df_summary_plot.png")
    plt.close()

    shap_df = pd.DataFrame(data=shap_values, columns=test_df.drop(columns=["y"]).columns, index=test_df.index)

    closure_shap_mean = shap_df[test_df['Closed'] == 1].mean()
    continuous_shap_mean = shap_df[test_df['Closed'] == 0].mean()


    # 평균 차이 계산
    shap_diff = closure_shap_mean - continuous_shap_mean
    plt.rcParams['font.family'] = 'DejaVu Sans'
    shap_diff.sort_values(ascending=False).plot(kind='barh', xlabel='SHAP Difference', ylabel='Feature', figsize=(8,16))
    plt.tight_layout()
    plt.savefig("images/shap_difference.png")
    plt.close()

    ##################################################### 탁 추가

    #print("test_x shape:", test_x.shape)
    #condition = test_x[:, -1] == 0
    #print("Condition shape:", condition.shape)
    #print("Condition values:", condition)

    #filtered_test_x = test_x[condition, :]
    #print("Filtered test_x shape:", filtered_test_x.shape)

    #group_0 = test_df[test_df["Closed"] == 0]  # 지속 격자
    #group_1 = test_df[test_df["Closed"] == 1]  # 폐업 격자
    # shap_values_0 = explainer.shap_values(test_x[test_x[:,-1]==0,:])
    # plt.figure()
    # shap.summary_plot(shap_values_0, group_0.drop(columns=["y"]), show=False)
    # plt.savefig("shap_keep_df_summary_plot.png")
    # plt.close()

    # plt.figure()
    # shap_values_1 = explainer.shap_values(test_x[test_x[:,-1]==1,:])
    # shap.summary_plot(shap_values_1, group_1.drop(columns=["y"]), show=False)
    # plt.savefig("shap_paeup_df_summary_plot.png")
    # plt.close()


if __name__=="__main__":
    cate_mask =0

    keep_grid = ['다사60aa49bb' # 뚝섬역
                 , 
                 '다사60ba49ab' # 가죽거리
                 ]
    
    paeup_grid = ['다사60ba49bb' # 정비소
                 , 
                  '다사60ba48bb' # 뚝도시장
                  ]

    
    #visualize_raw_df(cate_mask = cate_mask ,paeup_grid = paeup_grid, keep_grid= keep_grid)
    
    """
    
   
    raw_df = pd.read_csv("card_category_conditioned_processed.csv", encoding='cp949', index_col=0)
    raw_df = raw_df[~pd.isna(raw_df['age'])]
    raw_df['date'] = pd.DatetimeIndex(raw_df['date'])
    #raw_df.set_index('date', inplace=True)

    raw_df.loc[:, '폐업'] = '중립'
    raw_df.loc[raw_df.code250.isin(paeup_grid), '폐업'] = '폐업'
    raw_df.loc[raw_df.code250.isin(keep_grid), '폐업'] = '지속'
    cafe_raw = raw_df[raw_df['cate_mask'] == cate_mask]
    cafe_raw = cafe_raw[cafe_raw['code250'] == '다사60ba49bb']
    #print(cafe_raw.columns)

    cafe_raw['date'] = pd.to_datetime(cafe_raw['date'])

    # Extract the day of the week (e.g., Monday, Tuesday, ...)
    cafe_raw['day_of_week'] = cafe_raw['date'].dt.day_name()
    target_variable = 'sum_use_count'

    # Group data by day of the week and calculate the mean (or sum)
    daywise_data = cafe_raw.groupby('day_of_week')[target_variable].mean()  # Use sum() for total per day

    # Sort days to ensure correct order
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    #ordered_days = ['월', '화', '수', '목', '금', '토', '일']
    daywise_data = daywise_data.reindex(ordered_days)

    # Plot the data
    plt.figure(figsize=(10, 6))
    daywise_data.plot(kind='bar', color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('요일')
    plt.ylabel(f'{target_variable}')  # Change to 'Total Sum Amount' if using sum()
    plt.title('정비소 (폐업) 평균 매출 건수')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    """

    #target_variable = 'y'
    #target_scope = 'grid'

    #visualize_grid_paeup_distribution(train_test_df1, target_variable, target_scope)

    #visualize_grid_paeup_bar_distribution(train_test_df1, target_variable)
    """
    sum_male_columns = [column for column in train_test_df1.columns if column.startswith("Count Male")]
    sum_feml_columns = [column for column in train_test_df1.columns if column.startswith("Count Female")]
    target_scope = 'grid'

    for target_variable in sum_male_columns:
        visualize_grid_paeup_distribution(train_test_df1, target_variable, target_scope)

    for target_variable in sum_feml_columns:
        visualize_grid_paeup_distribution(train_test_df1, target_variable, target_scope)
    """
    #target_grid = '다사60ba49bb'
    #visualize_within_grid(train_test_df1, target_grid, ratio_flag = True)

    #train_test_df1 = train_test_df1.loc[train_test_df1['Closed'] == 1] # 폐업일때
    #train_test_df1 = train_test_df1.loc[train_test_df1['Closed'] == 0] # 지속일때

    train_test_df1 = main(0, keep_grid = keep_grid, paeup_grid = paeup_grid)
    _, test_df, train_x, train_y, test_x, test_y = split_train_test(train_test_df1)
    fitted_model = train(train_x, train_y)
    y_pred = fitted_model.predict(test_x)
    
    print("keep")
    get_evaluations(test_y, y_pred)
    r2_test = r2_score(test_y, y_pred)
    r2_train = fitted_model.score(train_x, train_y) 

    r2 = r2_score(np.exp(test_y)-1, np.exp(y_pred)-1)

    r2_train = fitted_model.score(train_x,train_y)

    print("r2 test: ", r2_test)
    print("r2 train: ", r2_train)

    get_dohyeon_evaluation(test_df, paeup_grid=paeup_grid, keep_grid=keep_grid, fitted_model = fitted_model
                           , y_pred = y_pred, test_y = test_y, train_x = train_x, train_y = train_y)

    #get_shap(fitted_model, test_df, test_x)
    
    #visualize_within_grid()
    
    #visualize_raw_df(cate_mask, paeup_grid, keep_grid, target_variable= 'sum_total_cnt')
    