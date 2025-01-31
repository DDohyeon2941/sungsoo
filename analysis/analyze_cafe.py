# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:22:19 2024

@author: dohyeon
"""
import sys
import os

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
plt.rcParams['font.family'] = 'DejaVu Sans'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

from preprocess.preprocess_for_training  import main

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


def train(train_x, train_y):

    rf_model = RandomForestRegressor(random_state=42, max_features='sqrt', n_estimators=500)

    rf_model.fit(train_x, train_y)

    return rf_model

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
    plt.show()

def get_smape(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    return np.sum(np.nan_to_num((np.abs(pred_y-real_y)) / ((real_y+pred_y)/2),0))/real_y.shape[0]



def find_both_non_zero_idx(real_y, pred_y):
    return np.setdiff1d(np.arange(real_y.shape[0]), np.intersect1d(np.where(real_y==0)[0], np.where(pred_y==0)[0]))

#%%
if __name__=="__main__":

    moving_dir_dict = {'2023':r'..\preprocess\moving\sungsoo_grid_arrival_2023_20241114.csv',
                       '2024':r'..\preprocess\moving\sungsoo_grid_arrival_2024_20241114.csv',
                       '20231':r'..\preprocess\moving\sungsoo_grid_arrival_2023_20241125.csv',
                       '20241':r'..\preprocess\moving\sungsoo_grid_arrival_2024_20241125.csv'}
    train_test_df1 = main(2,
                          keep_grid= ['다사59bb49ba'],
                          paeup_grid = ['다사60ab50aa', '다사60ab49ab'],
                          saup_dir=r'..\preprocess\paeup\saup_number.csv',
                          sales_dir=r'..\preprocess\sales\card_category_conditioned_processed.csv',
                          weather_dir=r'..\preprocess\weather\weather.csv',
                          transport_dir=r'..\preprocess\transport\sungsoo_prep_transport_by_dohyeon_20241115.csv',
                          moving_dir_dict=moving_dir_dict)
    #train_test_df1 = main(2, keep_grid= ['다사59bb49ba'], paeup_grid = ['다사60ab49ab'])

    train_df, test_df, train_x, train_y, test_x, test_y = split_train_test(train_test_df1)

    fitted_model = train(train_x, train_y)
    #%%
    y_pred = fitted_model.predict(test_x)


    test_df1 = pd.DataFrame(index=test_df.index)
    test_df1.loc[:,'pred_y'] = y_pred

    paeup_real_y = np.exp(test_df['y'].unstack()[['다사60ab49ab',  '다사60ab50aa']].stack().values)-1
    paeup_pred_y = np.exp(test_df1.unstack()['pred_y'][['다사60ab49ab',  '다사60ab50aa']].stack().values)-1

    keep_real_y = np.exp(test_df['y'].unstack()['다사59bb49ba'].values)-1
    keep_pred_y = np.exp(test_df1.unstack()['pred_y']['다사59bb49ba'].values)-1



#%%



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



    #%%

    r2 = r2_score(np.exp(test_y)-1, np.exp(y_pred)-1)

    r2_train = fitted_model.score(train_x,train_y)

    print(f"R2 Score: {r2:.2f}")
    print(f"R2 Score1: {r2_train:.2f}")

    visualize_importance(fitted_model, test_df)

    explainer = shap.TreeExplainer(fitted_model)
    
    # 테스트 데이터에 대해 SHAP 값 계산
    #%%
    shap_values = explainer.shap_values(test_x)
    
    shap.summary_plot(shap_values, test_df[test_df.columns[:-1]])
    
    shap_df = pd.DataFrame(data=shap_values, columns=test_df.drop(columns=["y"]).columns, index=test_df.index)

    #%% cal shap_diff


    closure_shap_mean = shap_df[test_df['Closed'] == 1].mean()
    continuous_shap_mean = shap_df[test_df['Closed'] == 0].mean()
    
    # 평균 차이 계산
    shap_diff = closure_shap_mean - continuous_shap_mean
    shap_diff_df= pd.concat([closure_shap_mean, continuous_shap_mean], axis=1)
    shap_diff_df.columns = ['Closed','Open']
    shap_diff_df.plot(kind='bar', figsize=(20,8), ylabel='SHAP', xlabel='Feature')
    print(shap_diff.sort_values(ascending=False))
    plt.rcParams['font.family'] = 'DejaVu Sans'
    shap_diff.sort_values(ascending=False).plot(kind='barh', xlabel='SHAP Difference', ylabel='Feature', figsize=(8,16))

    #%% 업무지구
    plt.rcParams['font.family'] = 'NanumGothic'  # 또는 'Malgun Gothic', 'AppleGothic' 등 사용 가능한 글꼴 지정
    
    # 음수 기호를 위해 유니코드 설정 유지
    plt.rcParams['axes.unicode_minus'] = False  # 음수 기호가 깨지는 문제 해결
    shap_df['Count Female 30-39'].unstack().plot()
    shap_df['Count Female 20-29'].unstack().plot()
    shap_df['Count Male 30-39'].unstack().plot()
    shap_df['Count Male 20-29'].unstack().plot()

    train_df['y'].unstack().plot()

    #%%
    s1 = shap_df['Count Female 30-39'].unstack()
    s1.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']

    colors = {"역세권 (지속)": "SkyBlue", "학교주변 (폐업)": "DarkRed", "업무지구 (폐업)": "LightCoral"}

    #%%

    s2 = shap_df['Count Male 30-39'].unstack()
    s2.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']



    #%%
    s3 = shap_df['Count Female 20-29'].unstack()
    s3.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']




    #%%
    s4 = shap_df['Count Male 20-29'].unstack()
    s4.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']



    #%%

    fig1, axes1 = plt.subplots(2,2, figsize=(12,12), sharey=True, sharex=True)

    flat_axes = axes1.flatten()

# 각 플롯에 데이터 추가 (예: 플롯 번호 표시)

    for column in s1.columns:
        for i, ax in enumerate(flat_axes):
            if i==0:
                ax.plot(s1.index, s1[column], label=column, color=colors[column])
                ax.set_title("30대 여성 유동량", fontsize=16)
            elif i==1:
                ax.plot(s2.index, s2[column], label=column, color=colors[column])
                ax.set_title("30대 남성 유동량", fontsize=16)
            elif i==2:
                ax.plot(s3.index, s3[column], label=column, color=colors[column])
                ax.set_title("20대 여성 유동량", fontsize=16)
            elif i==3:
                ax.plot(s4.index, s4[column], label=column, color=colors[column])
                ax.set_title("20대 남성 유동량", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("SHAP", fontsize=12)
            ax.tick_params(axis='x', labelrotation=90)

            # 범례와 그리드
            ax.legend(title="격자 특성")
            ax.grid(True, linestyle="--", alpha=0.6)
    #%%

    s5 = train_df['y'].unstack()
    s5.columns =  ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']

    # 그래프 생성
    plt.figure(figsize=(12, 6))
    
    for column in s5.columns:
        plt.plot(s5.index, s5[column], label=column, color=colors[column])
    plt.title("2023년", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("매출액", fontsize=12)
    
    # 범례와 그리드
    plt.legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.23),ncols=3)
    plt.grid(True, linestyle="--", alpha=0.6)


    #%%

    s6 = test_df['y'].unstack()
    s6.columns =  ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']

    # 그래프 생성
    plt.figure(figsize=(12, 6))
    
    for column in s6.columns:
        plt.plot(s6.index, s6[column], label=column, color=colors[column])
    plt.title("2024년", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("매출액", fontsize=12)
    
    # 범례와 그리드
    plt.legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.23),ncols=3)
    plt.grid(True, linestyle="--", alpha=0.6)




    #%% 분포찍어보기
    target_cols = ['Count Male 30-39', 'Count Female 30-39', 'Count Male 20-29', 'Count Female 20-29',
                   'Influx Ratio Female 30-39']


    t1 = train_df['Count Female 30-39'].unstack()
    t1.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']
    t1 = t1[['역세권 (지속)', '학교주변 (폐업)']]
    colors = {"역세권 (지속)": "SkyBlue", "학교주변 (폐업)": "DarkRed"}

    plt.figure(figsize=(12, 6))
    
    for column in t1.columns:
        plt.plot(t1.index, t1[column], label=column, color=colors[column])
    plt.title("2023년 30대 여성 유동량", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    #plt.xticks(rotation=90)
    plt.legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.23),ncols=2)

    #%%
    t2 = train_df['Influx Ratio Female 30-39'].unstack()
    t2.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']
    t2 = t2[['역세권 (지속)', '학교주변 (폐업)']]

    plt.figure(figsize=(12, 6))
    
    for column in t2.columns:
        plt.plot(t2.index, t2[column], label=column, color=colors[column])
    plt.title("2023년 30대 여성 외부유입비중", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    #plt.xticks(rotation=90)
    plt.legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.23),ncols=2)

    #%%

    t3 = train_df['Count Female 20-29'].unstack()
    t3.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']
    t3 = t3[['역세권 (지속)', '학교주변 (폐업)']]


    plt.figure(figsize=(12, 6))
    
    for column in t3.columns:
        plt.plot(t3.index, t3[column], label=column, color=colors[column])
    plt.title("2023년 20대 여성 유동량", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    #plt.xticks(rotation=90)
    plt.legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.23),ncols=2)


    #%%

    t4 = train_df['Influx Ratio Female 20-29'].unstack()
    t4.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']
    t4 = t4[['역세권 (지속)', '학교주변 (폐업)']]

    plt.figure(figsize=(12, 6))
    
    for column in t4.columns:
        plt.plot(t4.index, t4[column], label=column, color=colors[column])
    plt.title("2023년 20대 여성 외부유입비중", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Ratio", fontsize=12)
    #plt.xticks(rotation=90)
    plt.legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.23),ncols=2)



    #%%

    #%%
    fig1, axes1 = plt.subplots(1,2, sharey=True, sharex=True, figsize=(20,8))
    axes1 = axes1.flatten()
    for column in t1.columns:
        axes1[0].plot(t1.index, t1[column], label=column, color=colors[column])
        axes1[0].set_title("2023년 30대 여성 유동량", fontsize=16)
        axes1[0].set_xlabel("Date", fontsize=12)
        axes1[0].set_ylabel("Count", fontsize=12)
        axes1[0].legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.2),ncols=2, fontsize=12)


    for column in t3.columns:
        axes1[1].plot(t3.index, t3[column], label=column, color=colors[column])
        axes1[1].set_title("2023년 20대 여성 유동량", fontsize=16)
        axes1[1].set_xlabel("Date", fontsize=12)
        axes1[1].set_ylabel("Count", fontsize=12)
        axes1[1].legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.2),ncols=2, fontsize=12)








    #%%
    fig1, axes1 = plt.subplots(1,2, sharey=False, sharex=True, figsize=(20,8))
    axes1 = axes1.flatten()
    for column in t2.columns:
        axes1[0].plot(t2.index, t2[column], label=column, color=colors[column])
        axes1[0].set_title("2023년 30대 여성 외부유입비중", fontsize=16)
        axes1[0].set_xlabel("Date", fontsize=12)
        axes1[0].set_ylabel("Ratio", fontsize=12)
        axes1[0].set_ylim(0.5,0.9)
        axes1[0].legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.2),ncols=2, fontsize=12)


    for column in t4.columns:
        axes1[1].plot(t4.index, t4[column], label=column, color=colors[column])
        axes1[1].set_title("2023년 20대 여성 외부유입비중", fontsize=16)
        axes1[1].set_xlabel("Date", fontsize=12)
        axes1[1].set_ylabel("Ratio", fontsize=12)
        axes1[1].set_ylim(0.5,0.9)
        axes1[1].legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.2),ncols=2, fontsize=12)


#%%
    t5 = train_df['y'].unstack()
    t5.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']
    t5 = t5[['역세권 (지속)', '학교주변 (폐업)']]

    plt.figure(figsize=(12, 6))
    
    for column in t5.columns:
        plt.plot(t5.index, t5[column], label=column, color=colors[column])
    plt.title("2023년 평균 이동 시간", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Time", fontsize=12)
    #plt.xticks(rotation=90)
    plt.legend(title="격자 특성", loc='lower center',  bbox_to_anchor=(0.5,-0.23),ncols=2)
    #%%
    t6 = train_df['Count Female 60-69'].unstack()
    t6.columns = ['역세권 (지속)', '학교주변 (폐업)', '업무지구 (폐업)']
    t6 = t6[['역세권 (지속)', '학교주변 (폐업)']]
    #%%
    plt.scatter(t6['역세권 (지속)'],t5['역세권 (지속)'] )

    #%%

    train_df['Influx Ratio Female 30-39'].unstack().plot()

    train_df['Influx Ratio Female 20-29'].unstack().plot()

    train_df['Count Female 30-39'].unstack().plot()
    train_df['Count Female 20-29'].unstack().plot()

    train_df['Influx Ratio Male 30-39'].unstack().plot()
    train_df['Influx Ratio Male 20-29'].unstack().plot()



    train_df['Count Male 30-39'].unstack().plot()
    train_df['Count Male 20-29'].unstack().plot()

    train_df['Count Female 30-39'].unstack().plot()
    train_df['Count Female 20-29'].unstack().plot()

    train_df['Influx Ratio Female 50-59'].unstack().plot()


    new_train_df = train_df.reset_index().set_index('level_0')

    #%%
    fig1, axes1 = plt.subplots(1,1)
    new_train_df.loc[new_train_df['level_1'] =='다사60ab50aa'].y.plot(ax=axes1, c='b',label='paeup')
    new_train_df.loc[new_train_df['level_1'] =='다사59bb49ba'].y.plot(ax=axes1, c='g',label='keep')
    plt.legend()
    #%%
    fig1, axes1 = plt.subplots(1,1)
    new_train_df.loc[new_train_df['level_1'] =='다사60ab49ab'].y.plot(ax=axes1, c='g',label='keep')


