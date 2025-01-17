# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:21:55 2025

@author: dohyeon
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


if __name__ == "__main__":
    paeup_ratio_df = pd.read_csv(r'paeup_ratio_added_20250115.csv')

    fig1, axes1 = plt.subplots(2,1, figsize=(12,6))
    paeup_ratio_df[['third_word','explicit_paeup_ratio']].groupby('third_word').mean()['explicit_paeup_ratio'].sort_values().plot(kind='bar', ax=axes1[0], xlabel='행정동',ylabel='페업률', title='명시적 폐업 비율')
    axes1[0].set_xticks([np.where(np.array(paeup_ratio_df[['third_word','explicit_paeup_ratio']].groupby('third_word').mean()['explicit_paeup_ratio'].sort_values().index)=='성수')[0][0]])

    paeup_ratio_df[['third_word','paeup_ratio']].groupby('third_word').mean()['paeup_ratio'].sort_values().plot(kind='bar', ax=axes1[1], xlabel='행정동',ylabel='페업률', title='통합적 폐업 비율')
    axes1[1].set_xticks([np.where(np.array(paeup_ratio_df[['third_word','paeup_ratio']].groupby('third_word').mean()['paeup_ratio'].sort_values().index)=='성수')[0][0]])

