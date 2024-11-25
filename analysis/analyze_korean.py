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
from sungsoo_preprocess_final import main

keep_grid = ['다사60aa49bb', '다사60ba49ab']
paeup_grid = ['다사60ba49bb', '다사60ba48bb']


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


if __name__=="__main__":


    train_test_df1 = main(0, keep_grid= keep_grid, paeup_grid = paeup_grid)
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
    


