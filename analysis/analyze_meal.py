# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:22:19 2024

@author: dohyeon
"""

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import shap
import platform

from sungsoo_preprocess_final import main

# OS에 따른 폰트 설정
if platform.system() == "Darwin":
    font_name = font_manager.FontProperties(
        fname="/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    ).get_name()
elif platform.system() == "Windows":
    font_name = font_manager.FontProperties(
        fname="c:/Windows/Fonts/malgun.ttf"
    ).get_name()
else:
    font_name = "DejaVu Sans"

rc("font", family=font_name)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def split_train_test(total_df):
    scaler = StandardScaler()

    train_df = total_df.loc["2023"]
    test_df = total_df.loc["2024"]

    train_x = train_df[train_df.columns[:-2]].values
    train_y = train_df.y.values

    test_x = test_df[test_df.columns[:-2]].values
    test_y = test_df.y.values

    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    train_x = np.concatenate([train_x, train_df.paeup.values.reshape(-1, 1)], axis=1)
    test_x = np.concatenate([test_x, test_df.paeup.values.reshape(-1, 1)], axis=1)

    return train_df, test_df, train_x, train_y, test_x, test_y


def train(train_x, train_y):
    best_param = {
        "rf": {
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "n_estimators": 100,
        },
        "xgb": {
            "n_estimators": 100,
            "learning_rate": 0.3,
            "max_depth": 6,
            "gamma": 0,
            "subsample": 1,
        },
        "lgbm": {
            "num_iterations": 100,
            "learning_rate": 0.1,
            "min_data_in_leaf": 20,
            "boost_from_average": True,
        },
    }

    # RF
    rf_model = RandomForestRegressor(random_state=42, **best_param["rf"])
    rf_model.fit(train_x, train_y)

    # XGB
    xgb_model = XGBRegressor(random_state=42, **best_param["xgb"])
    xgb_model.fit(train_x, train_y)

    # LGBM
    lgbm_model = LGBMRegressor(random_state=42, **best_param["lgbm"])
    lgbm_model.fit(train_x, train_y)

    return rf_model, xgb_model, lgbm_model


def visualize_importance(fitted_model, test_df):
    # 변수 중요도 추출
    importances = fitted_model.feature_importances_
    feature_names = test_df.columns[:-1]

    # 중요도 데이터프레임 생성
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    # 변수 중요도 시각화
    plt.figure(figsize=(12, 10))
    plt.bar(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
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

keep_grid=["다사59bb49bb", "다사60ba48bb"]
paeup_grid=["다사60ba49bb"]

# %%
if __name__ == "__main__":
    train_test_df1 = main(
        1, keep_grid=keep_grid, paeup_grid=paeup_grid
    )
    _, test_df, train_x, train_y, test_x, test_y = split_train_test(train_test_df1)

    rf_model, xgb_model, lgbm_model = train(train_x, train_y)

    # 그룹 이름과 데이터 매핑
    groups = {
        "Total": (test_x, test_y),
        "Keep": (test_x[test_x[:, -1]==0], test_y[test_x[:, -1]==0]),
        "Closed": (test_x[test_x[:, -1]==1], test_y[test_x[:, -1]==1]),
    }

    # 결과 저장
    results = []

    # 그룹별 계산
    for group_name, (group_x, group_y) in groups.items():
        # 예측
        y_pred_rf = rf_model.predict(group_x)
        y_pred_xgb = xgb_model.predict(group_x)
        y_pred_lgbm = lgbm_model.predict(group_x)

        # 원복
        conv_group_y = np.exp(group_y) - 1
        conv_pred_y_rf = np.exp(y_pred_rf) - 1
        conv_pred_y_xgb = np.exp(y_pred_xgb) - 1
        conv_pred_y_lgbm = np.exp(y_pred_lgbm) - 1

        # 지표 계산
        rmse_rf = np.sqrt(mean_squared_error(conv_group_y, conv_pred_y_rf))
        rmse_xgb = np.sqrt(mean_squared_error(conv_group_y, conv_pred_y_xgb))
        rmse_lgbm = np.sqrt(mean_squared_error(conv_group_y, conv_pred_y_lgbm))

        mae_rf = mean_absolute_error(conv_group_y, conv_pred_y_rf)
        mae_xgb = mean_absolute_error(conv_group_y, conv_pred_y_xgb)
        mae_lgbm = mean_absolute_error(conv_group_y, conv_pred_y_lgbm)

        # MAPE 때만 0값 삭제
        drop_0_idx = np.where(conv_group_y == 0)[0]
        temp_test_y = np.delete(conv_group_y, drop_0_idx, axis=0)
        temp_pred_y_rf = np.delete(conv_pred_y_rf, drop_0_idx, axis=0)
        temp_pred_y_xgb = np.delete(conv_pred_y_xgb, drop_0_idx, axis=0)
        temp_pred_y_lgmb = np.delete(conv_pred_y_lgbm, drop_0_idx, axis=0)

        mape_rf = mean_absolute_percentage_error(temp_test_y, temp_pred_y_rf)
        mape_xgb = mean_absolute_percentage_error(temp_test_y, temp_pred_y_xgb)
        mape_lgbm = mean_absolute_percentage_error(temp_test_y, temp_pred_y_lgmb)

        smape_rf = get_smape(temp_test_y, temp_pred_y_rf)
        smape_xgb = get_smape(temp_test_y, temp_pred_y_xgb)
        smape_lgbm = get_smape(temp_test_y, temp_pred_y_lgmb)

        r2_rf = r2_score(group_y, y_pred_rf)
        r2_xgb = r2_score(group_y, y_pred_xgb)
        r2_lgbm = r2_score(group_y, y_pred_lgbm)

        # 결과 출력
        print(f"=== {group_name} ===")
        print(f"Root Mean Squared Error: \n RF : {rmse_rf:.2f} \n XGB : {rmse_xgb:.2f} \n LGBM : {rmse_lgbm:.2f}")
        print(f"Mean Absolute Error: \n RF : {mae_rf:.2f} \n XGB : {mae_xgb:.2f} \n LGBM : {mae_lgbm:.2f}")
        print(f"Mean Absolute Percentage Error: \n RF : {mape_rf:.2f} \n XGB : {mape_xgb:.2f} \n LGBM : {mape_lgbm:.2f}")
        print(f"SMAPE: \n RF : {smape_rf:.2f} \n XGB : {smape_xgb:.2f} \n LGBM : {smape_lgbm:.2f}")
        print(f"R2 Score: \n RF : {r2_rf:.2f} \n XGB : {r2_xgb:.2f} \n LGBM : {r2_lgbm:.2f}")

        # 결과 저장
        results.append({
            "group": group_name,
            "rmse": {"RF": rmse_rf, "XGB": rmse_xgb, "LGBM": rmse_lgbm},
            "mae": {"RF": mae_rf, "XGB": mae_xgb, "LGBM": mae_lgbm},
            "mape": {"RF": mape_rf, "XGB": mape_xgb, "LGBM": mape_lgbm},
            "smape": {"RF": smape_rf, "XGB": smape_xgb, "LGBM": smape_lgbm},
            "r2": {"RF": r2_rf, "XGB": r2_xgb, "LGBM": r2_lgbm},
        })

        df_data = []
        for entry in results:
            group = entry['group']
            for metric, values in entry.items():
                if metric != 'group':
                    for model, value in values.items():
                        df_data.append([group, metric, model, round(value, 2)])

        df = pd.DataFrame(df_data, columns=['Group', 'Metric', 'Model', 'Value'])
        df_pivot = df.pivot(index=['Group', 'Metric'], columns='Model', values='Value')
        print(df_pivot)

    # 실제값과 예측값 시각화
    fitted_model = xgb_model
    y_pred_xgb = xgb_model.predict(test_x)

    fig, axs = plt.subplots(3, 1, figsize=(10, 15)) 

    # 첫 번째 플롯
    axs[0].plot(np.unique(test_df.index.get_level_values(0)), test_y[::3], linestyle='-', c='k', label="Actual")
    axs[0].plot(np.unique(test_df.index.get_level_values(0)), y_pred_xgb[::3], linestyle='--', c='b', label="pred")
    axs[0].set_title('다사59bb49bb (지속)', fontsize=14)

    # 두 번째 플롯
    axs[1].plot(np.unique(test_df.index.get_level_values(0)), test_y[1::3], linestyle='-', c='k', label="Actual")
    axs[1].plot(np.unique(test_df.index.get_level_values(0)), y_pred_xgb[1::3], linestyle='--', c='b', label="pred")
    axs[1].set_title('다사60ba48bb (지속)', fontsize=14)

    # 세 번째 플롯
    axs[2].plot(np.unique(test_df.index.get_level_values(0)), test_y[2::3], linestyle='-', c='k', label="Actual")
    axs[2].plot(np.unique(test_df.index.get_level_values(0)), y_pred_xgb[2::3], linestyle='--', c='b', label="pred")
    axs[2].set_title('다사60ba49bb (폐업)', fontsize=14)

    # 공통 설정
    for ax in axs:
        ax.tick_params(axis='x', rotation=45)  
        ax.set_xlabel('Date')
        ax.set_ylabel('Y Value')
        ax.legend(loc='lower right')

    plt.tight_layout() 
    plt.show()

    visualize_importance(fitted_model, test_df)
    explainer = shap.TreeExplainer(fitted_model)

    # 테스트 데이터에 대해 SHAP 값 계산
    shap_values = explainer.shap_values(test_x)

    shap.summary_plot(shap_values, test_df[test_df.columns[:-1]])

    #격자별
    group_0 = test_df[test_df["paeup"] == 0]  # 지속 격자
    group_1 = test_df[test_df["paeup"] == 1]  # 폐업 격자

    shap_values_0 = explainer.shap_values(test_x[test_x[:,-1]==0,:])
    plt.figure()
    shap.summary_plot(shap_values_0, group_0.drop(columns=["y"]), show=False)

    plt.figure()
    shap_values_1 = explainer.shap_values(test_x[test_x[:,-1]==1,:])
    shap.summary_plot(shap_values_1, group_1.drop(columns=["y"]), show=False)

    shap_df = pd.DataFrame(
        data=shap_values,
        columns=test_df.drop(columns=["y"]).columns,
        index=test_df.index,
    )
    
    # cal shap_diff
    closure_shap_mean = shap_df[test_df["paeup"] == 1].mean()
    continuous_shap_mean = shap_df[test_df["paeup"] == 0].mean()

    # 평균 차이 계산
    shap_diff = closure_shap_mean - continuous_shap_mean
    print(shap_diff.sort_values(ascending=False))

# %% 시각화 용으로 변경
new_train_test_df1 = train_test_df1.reset_index()
new_train_test_df1 = new_train_test_df1.rename(columns = { 'level_0' : 'date' , 'level_1' : 'polygon_id1'})
new_train_test_df1 = new_train_test_df1.loc[new_train_test_df1['polygon_id1'].isin(keep_grid+paeup_grid)]
new_train_test_df1

# %% 세부 분석 01.
# 상위 20개의 변수 선택
mean_abs_shap_values = shap_df.abs().mean()
top_20_features = mean_abs_shap_values.nlargest(20).index.tolist()
top_20_features

# %% 유동량 검증
age_list = [
    "sum_male_00_19",
    "sum_male_20_29",
    "sum_male_30_39",
    "sum_male_40_49",
    "sum_male_50_59",
    "sum_male_60_69",
    "sum_male_70_79",
    "sum_male_80_89",
    "sum_feml_00_19",
    "sum_feml_20_29",
    "sum_feml_30_39",
    "sum_feml_40_49",
    "sum_feml_50_59",
    "sum_feml_60_69",
    "sum_feml_70_79",
    "sum_feml_80_89",
]
fig, axs = plt.subplots(8, 2, figsize=(15, 60))

for i in range(8):
    for j in range(2):
        data1 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사59bb49bb']
        data2 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사60ba48bb']
        data3 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사60ba49bb']
        
        col_name = age_list[i * 2 + j]
        axs[i, j].plot(data1['date'], data1[col_name], label='다사59bb49bb (지속)', c='MediumBlue', linestyle='-', alpha=0.5)
        axs[i, j].plot(data2['date'], data2[col_name], label='다사60ba48bb (지속)', c='SkyBlue', linestyle='--', alpha=0.5)
        axs[i, j].plot(data3['date'], data3[col_name], label='다사60ba49bb (폐업)', c='DarkRed', linestyle='-', alpha=0.5)
        
        axs[i, j].set_title(f"{age_list[i * 2 + j]}")
        axs[i, j].set_ylim(0, 1.5)
        axs[i, j].legend()
        if col_name in top_20_features:
            axs[i, j].grid(True)
        else:
            pass

plt.tight_layout()
plt.show()

# %%
move_list = [
    "sum_male_00_191",
    "sum_male_20_291",
    "sum_male_30_391",
    "sum_male_40_491",
    "sum_male_50_591",
    "sum_male_60_691",
    "sum_male_70_791",
    "sum_male_80_891",
    "sum_feml_00_191",
    "sum_feml_20_291",
    "sum_feml_30_391",
    "sum_feml_40_491",
    "sum_feml_50_591",
    "sum_feml_60_691",
    "sum_feml_70_791",
    "sum_feml_80_891",
]

fig, axs = plt.subplots(8, 2, figsize=(15, 60))

for i in range(8):
    for j in range(2):
        data1 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사59bb49bb']
        data2 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사60ba48bb']
        data3 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사60ba49bb']
        
        col_name = move_list[i * 2 + j]
        axs[i, j].plot(data1['date'], data1[col_name], label='다사59bb49bb (지속)', c='MediumBlue', linestyle='-', alpha=0.5)
        axs[i, j].plot(data2['date'], data2[col_name], label='다사60ba48bb (지속)', c='SkyBlue', linestyle='--', alpha=0.5)
        axs[i, j].plot(data3['date'], data3[col_name], label='다사60ba49bb (폐업)', c='DarkRed', linestyle='-', alpha=0.5)
        
        axs[i, j].set_title(f"{move_list[i * 2 + j]}")
        axs[i, j].set_ylim(0, 1.5)
        axs[i, j].legend()
        if col_name in top_20_features:
            axs[i, j].grid(True)
        else:
            pass

plt.tight_layout()
plt.show()

# %% 바플랏
# 데이터 필터링
filtered_data1 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사59bb49bb']
filtered_data2 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사60ba48bb']
filtered_data3 = new_train_test_df1[new_train_test_df1['polygon_id1'] == '다사60ba49bb']

mean_data_age_list = pd.DataFrame({
    '다사59bb49bb': filtered_data1[age_list].mean(),
    '다사60ba48bb': filtered_data2[age_list].mean(),
    '다사60ba49bb': filtered_data3[age_list].mean()
})

mean_data_age_list.plot(kind='bar', figsize=(10, 6))

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(3, 1, figsize=(10, 15)) 

# 첫 번째 플롯
axs[0].plot(filtered_data1['date'], filtered_data1['y'], linestyle='-', c='MediumBlue')
axs[0].set_title('다사59bb49bb (지속)', fontsize=14)

# 두 번째 플롯
axs[1].plot(filtered_data2['date'], filtered_data2['y'], linestyle='--', c='SkyBlue')
axs[1].set_title('다사60ba48bb (지속)', fontsize=14)

# 세 번째 플롯
axs[2].plot(filtered_data3['date'], filtered_data3['y'], linestyle='-', c='DarkRed')
axs[2].set_title('다사60ba49bb (폐업)', fontsize=14)

# 공통 설정
for ax in axs:
    ax.tick_params(axis='x', rotation=45)  
    ax.set_xlabel('Date')
    ax.set_ylabel('Y Value')

plt.tight_layout() 
plt.show()

# 0이 발생하는 요일 확인
print(filtered_data2[filtered_data2['y'] == 0]['date'].dt.day_name().value_counts())
print(filtered_data3[filtered_data3['y'] == 0]['date'].dt.day_name().value_counts())

# %% 면적
plt.figure(figsize=(10, 6))

plt.plot(filtered_data1['date'], filtered_data1['면적'], linestyle='-', c='MediumBlue', label='다사59bb49bb (지속)')
plt.plot(filtered_data2['date'], filtered_data2['면적'], linestyle='--', c='SkyBlue', label='다사60ba48bb (지속)')
plt.plot(filtered_data3['date'], filtered_data3['면적'], linestyle='-', c='DarkRed', label='다사60ba49bb (폐업)')

plt.title('면적', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('면적', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
mean_data_m = pd.DataFrame({
    'Polygon': ['다사59bb49bb', '다사60ba48bb', '다사60ba49bb'],
    'Mean': [
        filtered_data1['면적'].mean(),
        filtered_data2['면적'].mean(),
        filtered_data3['면적'].mean()
    ]
})
mean_data_m.set_index('Polygon', inplace=True)
mean_data_m.plot(kind='bar', figsize=(10, 6))

plt.xticks(rotation=45)
plt.tight_layout()
plt.xlabel("")
plt.show()

# %% 보증금
plt.figure(figsize=(10, 6))

plt.plot(filtered_data1['date'], filtered_data1['per_deposit'], linestyle='-', c='MediumBlue', label='다사59bb49bb (지속)')
plt.plot(filtered_data2['date'], filtered_data2['per_deposit'], linestyle='--', c='SkyBlue', label='다사60ba48bb (지속)')
plt.plot(filtered_data3['date'], filtered_data3['per_deposit'], linestyle='-', c='DarkRed', label='다사60ba49bb (폐업)')

plt.title('보증금', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('per_deposit', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
mean_data_b = pd.DataFrame({
    'Polygon': ['다사59bb49bb', '다사60ba48bb', '다사60ba49bb'],
    'Mean': [
        filtered_data1['per_deposit'].mean(),
        filtered_data2['per_deposit'].mean(),
        filtered_data3['per_deposit'].mean()
    ]
})
mean_data_b.set_index('Polygon', inplace=True)
mean_data_b.plot(kind='bar', figsize=(10, 6))

plt.xticks(rotation=45)
plt.tight_layout()
plt.xlabel("")
plt.show()

# %% 버스
plt.figure(figsize=(10, 6))

plt.plot(filtered_data1['date'], filtered_data1['bus_board'], linestyle='-', c='MediumBlue', label='다사59bb49bb (지속)')
plt.plot(filtered_data2['date'], filtered_data2['bus_board'], linestyle='--', c='SkyBlue', label='다사60ba48bb (지속)')
plt.plot(filtered_data3['date'], filtered_data3['bus_board'], linestyle='-', c='DarkRed', label='다사60ba49bb (폐업)')

plt.title('버스', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('bus_board', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
mean_data_b = pd.DataFrame({
    'Polygon': ['다사59bb49bb', '다사60ba48bb', '다사60ba49bb'],
    'Mean': [
        filtered_data1['bus_board'].mean(),
        filtered_data2['bus_board'].mean(),
        filtered_data3['bus_board'].mean()
    ]
})
mean_data_b.set_index('Polygon', inplace=True)
mean_data_b.plot(kind='bar', figsize=(10, 6))

plt.xticks(rotation=45)
plt.tight_layout()
plt.xlabel("")
plt.show()

# %% 지하철
plt.figure(figsize=(10, 6))

plt.plot(filtered_data1['date'], filtered_data1['subway_ratio'], linestyle='-', c='MediumBlue', label='다사59bb49bb (지속)')
plt.plot(filtered_data2['date'], filtered_data2['subway_ratio'], linestyle='--', c='SkyBlue', label='다사60ba48bb (지속)')
plt.plot(filtered_data3['date'], filtered_data3['subway_ratio'], linestyle='-', c='DarkRed', label='다사60ba49bb (폐업)')

plt.title('지하철', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('subway_ratio', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
mean_data_b = pd.DataFrame({
    'Polygon': ['다사59bb49bb', '다사60ba48bb', '다사60ba49bb'],
    'Mean': [
        filtered_data1['per_deposit'].mean(),
        filtered_data2['per_deposit'].mean(),
        filtered_data3['per_deposit'].mean()
    ]
})
mean_data_b.set_index('Polygon', inplace=True)
mean_data_b.plot(kind='bar', figsize=(10, 6))

plt.xticks(rotation=45)
plt.tight_layout()
plt.xlabel("")
plt.show()

# %% 이동시간
plt.figure(figsize=(10, 6))

plt.plot(filtered_data1['date'], filtered_data1['avg_move_time1'], linestyle='-', c='MediumBlue', label='다사59bb49bb (지속)')
plt.plot(filtered_data2['date'], filtered_data2['avg_move_time1'], linestyle='--', c='SkyBlue', label='다사60ba48bb (지속)')
plt.plot(filtered_data3['date'], filtered_data3['avg_move_time1'], linestyle='-', c='DarkRed', label='다사60ba49bb (폐업)')

plt.title('이동시간', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('avg_move_time1', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
mean_data_b = pd.DataFrame({
    'Polygon': ['다사59bb49bb', '다사60ba48bb', '다사60ba49bb'],
    'Mean': [
        filtered_data1['per_deposit'].mean(),
        filtered_data2['per_deposit'].mean(),
        filtered_data3['per_deposit'].mean()
    ]
})
mean_data_b.set_index('Polygon', inplace=True)
mean_data_b.plot(kind='bar', figsize=(10, 6))

plt.xticks(rotation=45)
plt.tight_layout()
plt.xlabel("")
plt.show()

#%%