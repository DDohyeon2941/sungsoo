# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:13:48 2025

@author: dohyeon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib import font_manager, rc
import fiona
from shapely.wkt import loads


def read_shp(shp_path):
    """single f"""
    f_list = []
    with fiona.open(shp_path) as src:
        for f in src:
            f_list.append(f)
    return f_list

def masking_group(xx):
    if xx in gr1_names:
        return 0
    elif xx in gr2_names:
        return 1
    elif xx in gr3_names:
        return 2
    elif xx in gr4_names:
        return 3
    else:
        return 4


if __name__ == "__main__":
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)

    # road data
    temp_df = pd.read_csv(r'whole_data.csv')
    code_li = temp_df['code250'].unique()
    """sale 기준, 업종 80개, 격자 60개"""
    #%%
    """1차 필터링"""
    #%%
    # 격자
    ## 격자별로, 특정 카테고리에 지출횟수가 없으면 1 누적
    code_zero_cate_arr = np.array([(temp_df.loc[temp_df['code250'] ==  uni_code].groupby('category').sum()['sum_use_count'] == 0).sum() for uni_code in code_li])

    ## 격자별로 단일한 카테고리에 대한 지출기록만 있으면 필터링 (격자 1차 필터링)
    print(temp_df.loc[temp_df['code250'].isin(code_li[np.where(code_zero_cate_arr==79)])].groupby(['code250']).sum()['sum_use_count'].index)
    
    ## 1차 필터링 대상 격자 (3개 필터링)
    ['다사58bb49ba', '다사59ba50ab', '다사60ba48ab']



    # 업종 (18개 필터링)
    new_temp_df = temp_df.loc[temp_df['sum_use_count'] > 0].groupby(['date','category','code250']).count()['sum_use_count'].unstack(fill_value=0)

    avg_df = new_temp_df.sum(axis=1).unstack().mean()


    ## 1차필터링 (업종 및 격자, 업종의 경우, 평균 일별 평균 매출이 발생한 격자의 수가 2보다 낮은 경우 분류가 불가능하므로 필터링 )
    temp_df1=temp_df.loc[(temp_df['category'].isin(avg_df[avg_df>=2].index.values))&(temp_df['code250'].isin(np.setdiff1d(code_li, ['다사58bb49ba', '다사59ba50ab', '다사60ba48ab'])))].reset_index(drop=True)

    #%%
    """2차 필터링"""
    #%%
    # 서울시 격자 shp 불러오기
    shp_list = read_shp(r'D:\project_repository\competition\성동구_20240705\국토지리_행안부_250격자 코드 match\match.shp')

    # 1차 필터링된 상태에서의 격자명 리스트업
    code_li1 = temp_df1['code250'].unique()

    # 격자명을 기반으로 Polygon index 추출
    shp_idx_list = [xidx for xidx, xx in enumerate(shp_list) if xx['properties']['GID'] in code_li1]
    shp_idx_list1 = [xidx for xidx, xx in enumerate(shp_list) if xx['properties']['GID'] in code_li]

    """격자 베이스 정보 추출 (code250, GID, Polygon) 격자 총 60개"""
    pd.DataFrame(data=[[shp_list[xx]['properties']['GID'], shp_list[xx]['properties']['CELL_ID'], Polygon(shp_list[xx]['geometry']['coordinates'][0]).wkt] for xx in shp_idx_list1]).to_csv(r'sungsoo_match_60_polygon.csv')

    # 1차 필터링된 격자 shp 파일 추출
    pd.DataFrame(data=[[shp_list[xx]['properties']['GID'], Polygon(shp_list[xx]['geometry']['coordinates'][0]).wkt] for xx in shp_idx_list]).to_csv(r'sungsoo_match_57_polygon.csv')
    #%%
    """1차 필터링된 격자를 기반, 매출액과 매출건수를 기반해 히트맵 그리기"""
    # 매출액
    pd.merge(temp_df1.groupby(['code250']).sum()['sum_amount'].loc[code_li1],
    pd.DataFrame(data=[[shp_list[xx]['properties']['GID'], Polygon(shp_list[xx]['geometry']['coordinates'][0]).centroid.wkt] for xx in shp_idx_list]), left_on = 'code250', right_on=0)[[0,1,'sum_amount']].to_csv(r'sungsoo_match_57_amount_heatmap.csv', index=False)

    # 매출건수
    pd.merge(temp_df1.groupby(['code250']).sum()['sum_use_count'].loc[code_li1],
    pd.DataFrame(data=[[shp_list[xx]['properties']['GID'], Polygon(shp_list[xx]['geometry']['coordinates'][0]).centroid.wkt] for xx in shp_idx_list]), left_on = 'code250', right_on=0)[[0,1,'sum_use_count']].to_csv(r'sungsoo_match_57_use_count_heatmap.csv', index=False)


    #%% 히트맵을 기반해, 격자 2차 필터링
    
    match_57_polygon_df = pd.DataFrame(data=[[shp_list[xx]['properties']['GID'],shp_list[xx]['properties']['CELL_ID'], Polygon(shp_list[xx]['geometry']['coordinates'][0]).wkt] for xx in shp_idx_list])

    # 격자 필터링 (12개)
    drop_idx = np.sort(np.array([42, 36, 29, 53, 54, 50, 49, 41, 26, 22, 15, 12]))
    drop_code_li = np.array(['다사60bb48ab', '다사60bb48ba', '다사60ba48ba', '다사60ab48ba',
           '다사60ab50ab', '다사60aa50ab', '다사60aa48ba', '다사59bb50ab',
           '다사59ba49aa', '다사59ba49ab', '다사59ba50aa', '다사59ab49ba',
           '다사58bb49ba', '다사59ba50ab', '다사60ba48ab'],
          dtype=object)
    
    # 2차 필터링된 격자와 선택된 격자에 대해서 각각 polygon 생성
    match_57_polygon_df.loc[drop_idx].reset_index(drop=True).to_csv(r'sungsoo_2nd_drop_polygons_12.csv')
    match_57_polygon_df.loc[np.setdiff1d(np.arange(57), drop_idx)].reset_index(drop=True).to_csv(r'sungsoo_2nd_drop_polygons_45.csv')
    
    # 격자에 대해서 2차 필터링된 데이터 생성
    temp_df2 = temp_df1.loc[temp_df1['code250'].isin(np.setdiff1d(code_li1, drop_code_li))].reset_index(drop=True)

    # 업종에 대해서 카테고리를 정할 수 없는 경우 필터링
    temp_df2 = temp_df2.loc[temp_df2['category'] != 'ZZ_나머지'].reset_index(drop=True)
    
    # 업종, 레퍼런스 기반해, 대상 업종 선정 (대분류 4가지를 제외하고는 모두 제외)

    gr1_names = ['한식', '일식', '양식', '중식', '제과점', '커피전문점', '패스트푸드', '기타요식']
    gr2_names = [
    '슈퍼마켓 일반형','편의점','정육점','농수산물','주류판매','기타식품',
    '의복/의류','패션잡화','시계/귀금속','헬스장',
    '실내/실외골프장','스포츠시설','서점','영화/공연','게임방/오락실',
    '스포츠/레저용품','문화용품','화원',
    '완구/아동용자전거','애완동물','가전','가구',
    '인테리어/건축자재/주방기구','사무기기/문구용품',
    '컴퓨터/소프트웨어']
    
    gr3_names =['모텔,여관,기타숙박',
    '미용실',
    '화장품',
    '싸우나/목욕탕',
    '세탁소',
    '예식장/결혼서비스',
    '회계/변리서비스',
    '연구/번역서비스',
    '학원/학습지',
    '일반병원',
    '치과병원',
    '동물병원',
    '한의원',
    '기타의료']
    
    gr4_names =['수입자동차',
    '오토바이',
    '자동차용품',
    '자동차서비스',
    '주차장']
    


    temp_df2.loc[:,'group'] = [masking_group(xx) for xx in temp_df2['category']]
    
    
    temp_df3 = temp_df2.loc[temp_df2['group']!=4].reset_index(drop=True)

    ## sale 데이터에서 격자 및 업종에대해서 1,2차 필터링 후 최종 저장
    temp_df3.to_csv(r'whole_data_45_grid_51_cate.csv', index=False)

    
    
    














