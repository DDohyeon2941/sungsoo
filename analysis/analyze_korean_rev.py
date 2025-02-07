# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:22:19 2024

@author: dohyeon
"""
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import shap
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

from preprocess.preprocess_for_training  import main
grid_name_mapping = {
    '다사60aa49bb': '역세권 (지속)',   # 뚝섬역
    '다사60ba49ab': '가죽거리 (지속)',   # 가죽거리
    '다사60ba49bb': '정비소 (폐업)',    # 정비소
    '다사60ba48bb': '뚝도시장 (폐업)', # 뚝도시장

}

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
    #plt.savefig('Feature_Importance.png')
    #plt.show()
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

    raw_df = pd.read_csv(r'..\preprocess\sales\card_category_conditioned_processed.csv', encoding='cp949', index_col=0)
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

    sns.lineplot(data=korean_raw, x='date', y='sum_amount', hue='code250', palette=grid_color_mapping)
    plt.xlabel('Date')
    plt.ylabel(f'{target_variable}')
    plt.legend(title = '격자 특성')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig(f'images/{target_variable}.png')
    plt.show()
    plt.close()

def get_shap(fitted_model, test_df, test_x):
    visualize_importance(fitted_model, test_df)

    explainer = shap.TreeExplainer(fitted_model)
    
    # 테스트 데이터에 대해 SHAP 값 계산

    shap_values = explainer.shap_values(test_x)
    plt.figure()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    shap.summary_plot(shap_values, test_df[test_df.columns[:-1]], show=False)
    plt.tight_layout()
    plt.show()
    #plt.savefig("images/shap_test_df_summary_plot.png")
    plt.close()

    shap_df = pd.DataFrame(data=shap_values, columns=test_df.drop(columns=["y"]).columns, index=test_df.index)

    closure_shap_mean = shap_df[test_df['Closed'] == 1].mean()
    continuous_shap_mean = shap_df[test_df['Closed'] == 0].mean()


    # 평균 차이 계산
    shap_diff = closure_shap_mean - continuous_shap_mean
    plt.rcParams['font.family'] = 'DejaVu Sans'
    shap_diff.sort_values(ascending=False).plot(kind='barh', xlabel='SHAP Difference', ylabel='Feature', figsize=(8,16))
    plt.tight_layout()
    plt.show()
    #plt.savefig("images/shap_difference.png")
    plt.close()

#%%
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

    moving_dir_dict = {'2023':r'..\preprocess\moving\sungsoo_grid_arrival_2023_20241114.csv',
                       '2024':r'..\preprocess\moving\sungsoo_grid_arrival_2024_20241114.csv',
                       '20231':r'..\preprocess\moving\sungsoo_grid_arrival_2023_20241125.csv',
                       '20241':r'..\preprocess\moving\sungsoo_grid_arrival_2024_20241125.csv'}

    train_test_df1 = main(0,
                          keep_grid= keep_grid,
                          paeup_grid = paeup_grid,
                          saup_dir=r'..\preprocess\paeup\saup_number.csv',
                          sales_dir=r'..\preprocess\sales\card_category_conditioned_processed.csv',
                          weather_dir=r'..\preprocess\weather\weather.csv',
                          transport_dir=r'..\preprocess\transport\sungsoo_prep_transport_by_dohyeon_20241115.csv',
                          moving_dir_dict=moving_dir_dict)
  
    _, test_df, train_x, train_y, test_x, test_y = split_train_test(train_test_df1)
    fitted_model = train(train_x, train_y)
    y_pred = fitted_model.predict(test_x)
    
    get_evaluations(test_y, y_pred)

    get_dohyeon_evaluation(test_df, paeup_grid=paeup_grid, keep_grid=keep_grid, fitted_model = fitted_model
                           , y_pred = y_pred, test_y = test_y, train_x = train_x, train_y = train_y)

    get_shap(fitted_model, test_df, test_x)
    
    visualize_raw_df(cate_mask, paeup_grid, keep_grid, target_variable= 'sum_amount')
    