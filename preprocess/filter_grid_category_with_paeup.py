# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:58:19 2025

@author: dohyeon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pyproj import Transformer
from shapely.geometry import Point
from shapely import from_wkt
import seaborn as sns
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

def visualize_paeup_distribution(paeup_df, cate_name, gr_idx):

    if gr_idx ==4:
        sel_df = paeup_df.loc[paeup_df['cate'].isin(cate_name)]
    else:
        sel_df = paeup_df.loc[paeup_df['opnSvcNm'].isin(cate_name)]

    test_t1 = sel_df.loc[sel_df['paeup']==1]
    
    test_t1.loc[:,'date'] = pd.to_datetime(test_t1['dcbYmd'])
    test_t1.loc[:, 'year'] = [xx.year for xx in test_t1['date']]

    if gr_idx in [1,4]:
        fig1, axes1 = plt.subplots(1,1, figsize=(12,8))
        sns.heatmap(test_t1[['uptaeNm','year','paeup']].groupby(['uptaeNm','year']).sum()['paeup'].unstack(fill_value=0), cmap='rainbow',ax=axes1, vmax=140)

    else:
        fig1, axes1 = plt.subplots(1,1, figsize=(12,8))
        sns.heatmap(test_t1[['opnSvcNm','year','paeup']].groupby(['opnSvcNm','year']).sum()['paeup'].unstack(fill_value=0), cmap='rainbow',ax=axes1, vmax=140)


def mask_category(paeup_df, cate_name, jname_df, gr_idx):

    paeup_df.loc[paeup_df.uptaeNm.isin(cate_name),'cate_mask'] = gr_idx
    paeup_df.loc[paeup_df.bplcNm.isin(jname_df.loc[jname_df['1']==gr_idx]['0']), 'cate_mask'] = gr_idx

    return paeup_df


def mask_grid(paeup_df, shp_df):

    grid_info_li = []
    for xidx, xx in enumerate(paeup_df['pnt']):
        start_len = len(grid_info_li)
        for yy in shp_df[['0','1','2']].values:
            if from_wkt(yy[2]).intersects(from_wkt(xx)):
                grid_info_li.append([yy[0], yy[1]])
        if len(grid_info_li)==start_len:
            grid_info_li.append([np.nan, np.nan])
    return grid_info_li


if __name__ == "__main__":
    """3차 필터링 (업종)"""
    # 폐업 데이터셋 불러오기
    temp_df = pd.read_csv(r'C:\Users\dohyeon\Downloads\store_status_seungsu.csv')

    # sales 데이터와 매칭이 되는 업종 선정
    unique_name = temp_df['opnSvcNm'].unique()
    
    filtered_name = ['의료기기수리업','사료제조업','게임물배급업','게임물제작업',
     '국제회의기획업','대중문화예술기획업', '비디오물배급업','비디오물제작업', '국내여행업','국내외여행업',
     '종합여행업', '영화배급업','영화수입업','영화제작업', '온라인음악서비스제공업', '음반.음악영상물배급업',
     '음반.음악영상물제작업', '옥외광고업', '위탁급식영업','집단급식소','집단급식소식품판매업', '축산가공업','식육포장처리업','식품소분업','식품운반업','식품제조가공업',
     '식품첨가물제조업','식품판매업(기타)','용기·포장지제조업','용기냉동기특정설비','유통전문판매업',
     '축산물운반업','단란주점영업','방문판매업','전화권유판매업','통신판매업', '목재수입유통업','계량기수리업',
     '계량기수입업','계량기제조업','계량기증명업','고압가스업','석유판매업','액화석유가스용품제조업체','특정고압가스업','지하수시공업체', '지하수영향조사기관', '지하수정화업체', '개인하수처리시설관리업(사업장)','건물위생관리업','건설폐기물처리업','단독정화조/오수처리시설설계시공업','분뇨수집운반업','소독업', '저수조청소업', '환경전문공사업','환경측정대행업','민방위급수시설','무료직업소개소','유료직업소개소' ]
    
    selected_name = ['병원', '의원', '안전상비의약품 판매업소', '약국', '의료유사업', '안경업', '의료기기판매(임대)업',
     '치과기공소','동물병원', '동물약국', '동물용의료용구판매업','동물용의약품도매상', '동물생산업',
     '동물판매업','동물전시업','동물위탁관리업', '동물미용업','동물운송업', '공연장','유원시설업(기타)',
     '노래연습장업','일반게임제공업', '청소년게임제공업','인터넷컴퓨터게임시설제공업',
     '관광숙박업','숙박업','외국인관광도시민박업','일반야영장업','영화상영관','영화상영업',
     '인쇄사','출판사','미용업','이용업','세탁업', '건강기능식품유통전문판매업','건강기능식품일반판매업',
     '제과점영업','즉석판매제조가공업','일반음식점','휴게음식점','대규모점포','배출가스전문정비사업자(확인검사대행자)','수질오염원설치시설(기타)', '골프연습장업', '당구장업','체육도장업','체력단련장업','담배도매업','담배소매업','목욕장업','대기오염물질배출시설설치사업장', '축산판매업','식품자동판매기업']
    
    
    selected_df = temp_df.loc[temp_df['opnSvcNm'].isin(selected_name)].reset_index(drop=True)

    selected_df.loc[:, 'cate'] = selected_df['uptaeNm']
    
    # 통신판매업 중 일부 업종 선택
    tong_df = temp_df.loc[temp_df['opnSvcNm'] == '통신판매업'].reset_index(drop=True)
    tong_df.loc[:, 'cate'] = [xx.split(' ')[0] for xx in tong_df['uptaeNm']]
    
    
    tong_df1 = tong_df.loc[tong_df['cate'].isin(['가구/수납용품', '가전', '건강/식품', '교육/도서/완구/오락', '레져/여행/공연', '의류/패션/잡화/뷰티', '자동차/자동차용품', '컴퓨터/사무용품'])].reset_index(drop=True)
    
    
    tong_df1.loc[:, 'new_cate'] = [0 if ('상품권' in xx )| ('종합몰' in xx) | ('성인/성인용품' in xx) | ('기타' in xx )else 1 for xx in tong_df1['uptaeNm'] ]
    
    
    tong_df2 = tong_df1.loc[tong_df1['new_cate'] ==1].reset_index(drop=True)
    
    
    final_df = pd.concat([selected_df, tong_df2[['siteWhlAddr', 'sitePostNo', 'trdStateGbn', 'x', 'rdnWhlAddr', 'dcbYmd','trdStateNm', 'y', 'opnSfTeamCode', 'apvPermYmd', 'rdnPostNo', 'dtlStateGbn', 'dtlStateNm', 'opnSvcNm', 'bplcNm', 'uptaeNm', 'cate']]]).reset_index(drop=True)
    
    
    #final_df.to_csv(r'sungsoo_paeup_20240906.csv', index=False)

    #%%
    """sales 데이터셋내 대분류내 중 또는 소분류 업종과 비슷한 업종을 선별하고, 구분함"""
    # 폐업 분포를 시각화함

    gr1=['일반음식점', '휴게음식점', '제과점영업',  '즉석판매제조가공업', ]
    gr2=['통신판매업', '건강기능식품유통전문판매업', '건강기능식품일반판매업', '축산판매업', '식품자동판매기업','안경업', '의료기기판매(임대)업','동물판매업','동물용의료용구판매업']
    gr3=['인터넷컴퓨터게임시설제공업','일반게임제공업', '청소년게임제공업', '공연장', '유원시설업(기타)', '노래연습장업',
         '골프연습장업', '당구장업', '체육도장업', '체력단련장업', '담배도매업', '담배소매업', '목욕장업','관광숙박업',
         '숙박업', '외국인관광도시민박업', '일반야영장업', '영화상영관', '영화상영업','병원', '의원', '안전상비의약품 판매업소', '약국', '의료유사업',
         '미용업', '이용업', '세탁업','동물전시업', '동물위탁관리업', '동물미용업', '동물운송업','동물병원', '동물약국',]
    
    gr4=['자동차 종합 수리업', '도금업','자동차 전문 수리업', '전자코일, 변성기 및 기타 전자유도자 제조업',
          '자동차 수리업','자동차 수리 및 세차업', '자동차 및 모터사이클 수리업', '타이어 및 튜브 제조업','자동차부품 제조업','자동차/자동차용품']

    final_df.loc[:, 'paeup'] = [1 if xx in(['폐업', '직권폐업','말소','등록취소','직권말소','허가취소','폐업처리','직권취소','폐쇄','지정취소']) else 0 for xx in final_df['dtlStateNm']]

    final_df = (final_df.loc[final_df['uptaeNm']!='편의점']).reset_index(drop=True)


    """대분류별로 폐업분포를 확인, 최종적으로 요식(Group1)만 사용하기로 함"""
    visualize_paeup_distribution(final_df, gr1, 1)
    visualize_paeup_distribution(final_df, gr2, 2)
    visualize_paeup_distribution(final_df, gr3, 3)
    visualize_paeup_distribution(final_df, gr4, 4)

    #%%
    """요식인 경우만 산출"""

    final_df.loc[final_df['opnSvcNm'] == '즉석판매제조가공업']

    final_df.loc[final_df['uptaeNm'] == '즉석판매제조가공업']

    final_df['uptaeNm'].unique()
    final_df['opnSvcNm'].unique()

    """요식인 경우만 인뎅싱"""
    new_final_df = (final_df.loc[final_df['opnSvcNm'].isin(gr1)]).reset_index(drop=True)

    new_final_df

    #%%
    """요식인 경우, 즉석판매제조가공업을 포함한 카테고리 마스킹"""

    transformer = Transformer.from_crs("EPSG:5174", "EPSG:5179", always_xy=True)

    new_final_df.loc[:,'pnt'] = [Point(transformer.transform(xx[0],xx[1])).wkt for xx in new_final_df[['x','y']].values]


    new_final_df.loc[:, 'end_date'] = pd.to_datetime(new_final_df['dcbYmd'])
    new_final_df.loc[:, 'start_date'] = pd.to_datetime(new_final_df['apvPermYmd'])
    
    new_final_df.loc[:, 'end_year'] = [xx.year for xx in new_final_df['end_date']]
    new_final_df.loc[:, 'end_month'] = [xx.month for xx in new_final_df['end_date']]
    
    new_final_df.loc[:, 'day_count'] = [xx.days for xx in (new_final_df['end_date'] - new_final_df['start_date'])]

    under_30_names = (new_final_df.loc[(new_final_df.day_count<30) & (new_final_df.uptaeNm=='즉석판매제조가공업')])['bplcNm'].unique()

    ###
    jfood_name = pd.read_csv(r'C:\Users\dohyeon\Downloads\sungsoo_store_name.csv')
    jfood_name = jfood_name.loc[~jfood_name['0'].isin(under_30_names)].reset_index(drop=True)

    new_final_df2 = new_final_df.loc[~((new_final_df['uptaeNm']=='즉석판매제조가공업') & (new_final_df['bplcNm'].isin(under_30_names)))].reset_index(drop=True)

    new_final_df2.loc[:, 'cate_mask'] = 5

    ### 요식업에 속하는 하위 카테고리 마스킹 (총 5개)
    new_final_df2 = mask_category(new_final_df2, ['김밥','김밥(도시락)','한식'], jfood_name, 0 )

    new_final_df2 = mask_category(new_final_df2, ['일식','경양식','중식','중국식'], jfood_name, 1)
    new_final_df2 = mask_category(new_final_df2, ['키즈까페','카페','라이브카페','제과점영업','다방','떡카페','아이스크림','전통찻집','커피숍','패스트푸드','통닭(치킨)'], jfood_name, 2)

    new_final_df2 = mask_category(new_final_df2, ['기타','호프/통닭','뷔페식','식육(숯불구이)','외국음식전문점(인도,태국등)','정종/대포집/소주방','출장조리','횟집','기타 휴게음식점',], jfood_name, 3)

    new_final_df2 = mask_category(new_final_df2, ['노래연습장업','감성주점'], jfood_name, 4)

    new_final_df2['cate_mask'].value_counts()

    #%%

    """격자정보 불러와서, 격자단위로 정의하기"""

    polygon_df = pd.read_csv(r'D:\project_repository\competition\성동구_20240705\modules\sungsoo_2nd_drop_polygons_45.csv',index_col=0)
    polygon_df

    new_final_df2.loc[:,['grid1','grid2']] = mask_grid(new_final_df2, polygon_df)

    final_final_df = pd.merge(left=new_final_df2, right=polygon_df, left_on='grid2',right_on='1').reset_index(drop=True)

    (final_final_df[['grid2','cate_mask','paeup']].groupby(['grid2','cate_mask']).sum()['paeup'].unstack() / final_final_df[['grid2','cate_mask','paeup']].groupby(['grid2','cate_mask']).count()['paeup'].unstack())[[0,1,2,3]].plot(rot=90)
        

    #%%
    t1 = final_final_df[['end_year','grid1','cate_mask','paeup']].loc[final_final_df['end_year']>2022].groupby(['grid1','cate_mask']).sum()['paeup'].unstack(fill_value=0)

    # 히트맵
    fig1, axes1 = plt.subplots(1,1, figsize=(12,8))
    sns.heatmap(t1[[0,1,2]], cmap='rainbow', ax=axes1, annot=True)
    axes1.set_xticks([0.5,1.5,2.5], labels=['한식','일식/양식/중식', '제과/커피/패스트푸드'])
    axes1.set_xlabel('중분류')
    axes1.set_ylabel('격자')
    axes1.set_title('격자별 중분류별 기간내 폐업수')
    
    # 바 그래프
    (t1[[0,1,2]].sum(axis=1)).sort_values(ascending=False).plot(kind='bar', xlabel='격자', ylabel='폐업수', title='격자별 기간내 폐업총수')
    

    #%%
    """요식업종에서 하위 업종을 재선택하는 과정에서 발생하는 폐업분포의 편차를 확인함"""

    new_final_df.loc[:,['grid1','grid2']] = mask_grid(new_final_df, polygon_df)


    arr1 = new_final_df[['grid2','paeup']].groupby(['grid2']).sum()['paeup'] / new_final_df.groupby(['grid2']).count()['paeup']
    arr2 = final_final_df[['grid2','paeup']].groupby(['grid2']).sum()['paeup'] / final_final_df.groupby(['grid2']).count()['paeup']
    
    fig1, axes1 = plt.subplots(1,1)
    arr1.plot(ax=axes1, rot=90, marker='o', xticks=np.arange(45), label='before')
    arr2.plot(ax=axes1, rot=90, marker='x', xticks=np.arange(45), label='after')
    
    axes1.legend()

    #%%

    final_final_df.to_csv(r'sungsoo_paeup_grid_coef_20250115.csv', index=False)













