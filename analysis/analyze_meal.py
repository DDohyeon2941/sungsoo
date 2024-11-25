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


if __name__ == "__main__":
    train_test_df1 = main(
        1, keep_grid=["다사59bb49bb", "다사60ba48bb"], paeup_grid=["다사60ba49bb"]
    )

    _, test_df, train_x, train_y, test_x, test_y = split_train_test(train_test_df1)

    rf_model, xgb_model, lgbm_model = train(train_x, train_y)

    y_pred_rf = rf_model.predict(test_x)
    y_pred_xgb = xgb_model.predict(test_x)
    y_pred_lgbm = lgbm_model.predict(test_x)

    mse_rf = mean_squared_error(test_y, y_pred_rf)
    mse_xgb = mean_squared_error(test_y, y_pred_xgb)
    mse_lgbm = mean_squared_error(test_y, y_pred_lgbm)
    
    r2_rf = r2_score(test_y, y_pred_rf)
    r2_xgb = r2_score(test_y, y_pred_xgb)
    r2_lgbm = r2_score(test_y, y_pred_lgbm)

    print(f"Mean Squared Error: \n RF : {mse_rf:.2f} \n XGB : {mse_xgb:.2f} \n LGBM : {mse_lgbm:.2f}")
    print(f"R2 Score: \n RF : {r2_rf:.2f} \n XGB : {r2_xgb:.2f} \n LGBM : {r2_lgbm:.2f} ")

    fitted_model = xgb_model
    
    visualize_importance(fitted_model, test_df)

    explainer = shap.TreeExplainer(fitted_model)

    # 테스트 데이터에 대해 SHAP 값 계산
    # %%
    shap_values = explainer.shap_values(test_x)

    shap.summary_plot(shap_values, test_df[test_df.columns[:-1]])

    shap_df = pd.DataFrame(
        data=shap_values,
        columns=test_df.drop(columns=["y"]).columns,
        index=test_df.index,
    )

    # %% cal shap_diff

    closure_shap_mean = shap_df[test_df["paeup"] == 1].mean()
    continuous_shap_mean = shap_df[test_df["paeup"] == 0].mean()

    # 평균 차이 계산
    shap_diff = closure_shap_mean - continuous_shap_mean
    print(shap_diff.sort_values(ascending=False))

    # %%
    shap_df["sum_male_60_69"].unstack().plot()
