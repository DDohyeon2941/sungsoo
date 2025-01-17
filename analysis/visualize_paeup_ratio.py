# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:21:55 2025

@author: dohyeon
"""
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":
    paeup_ratio_df = pd.read_csv(r'paeup_ratio_added.csv')

    fig1, axes1 = plt.subplots(2,1, figsize=(12,6))
    paeup_ratio_df[['third_word','explicit_paeup_ratio']].groupby('third_word').mean()['explicit_paeup_ratio'].sort_values().plot(kind='bar', ax=axes1[0], xlabel='행정동',ylabel='페업률', title='명시적 폐업 비율')
    axes1[0].set_xticks([130])

    paeup_ratio_df[['third_word','paeup_ratio']].groupby('third_word').mean()['paeup_ratio'].sort_values().plot(kind='bar', ax=axes1[1], xlabel='행정동',ylabel='페업률', title='통합적 폐업 비율')
    axes1[1].set_xticks([110])

