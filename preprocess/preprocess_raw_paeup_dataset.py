# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:44:06 2025

@author: dohyeon
"""

import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import re


def parse_xml_row(row, common_columns):
    """
        xml 파일 row 가져오기
        성동구 코드 opnSfTeamCode - 303000
    """
    row_data = {col: '' for col in common_columns}
    for child in row:
        if child.tag in common_columns:
            row_data[child.tag] = child.text.strip() if child.text else ''
    return row_data


def make_csv_file(zip_file_path):
    """
        XML 파일 -> csv 파일 변환
    """

    data_list = []

    common_columns = None

    with zipfile.ZipFile(zip_file_path, 'r') as z:
        xml_files = [file_name for file_name in z.namelist() if file_name.endswith('.xml')]
        
        for file_name in tqdm(xml_files, desc="Determining common columns"):
            with z.open(file_name) as f:
                tree = ET.parse(f)
                root = tree.getroot()
                
                columns = set()
                for header in root.iter('header'):
                    columns.update(child.tag for child in header.find('columns'))
                
                if common_columns is None:
                    common_columns = columns
                else:
                    common_columns.intersection_update(columns)

        for file_name in tqdm(xml_files, desc="Processing XML files"):
            with z.open(file_name) as f:
                tree = ET.parse(f)
                root = tree.getroot()

                for rows in root.iter('rows'):
                    
                    for row in tqdm(rows.findall('row'), desc="Processing rows", leave=False):
                        

                        row_data = parse_xml_row(row, common_columns)
                        if row_data is not None:
                            data_list.append(row_data)
        
    common_columns = list(common_columns)

    df = pd.DataFrame(data_list, columns=common_columns)

    return df

def filter_cols_under_2020(df):
    to_remove_columns = ['opnSvcId', 'mgtNo', 'apvCancelYmd', 'clgStdt', 'clgEnddt', 'ropnYmd', 'siteTel'
                        , 'siteArea', 'lastModTs', 'updateGbn', 'updateDt']

    df = df.drop(to_remove_columns ,axis = 1)

    df['dcbYmd'] = pd.to_datetime(df['dcbYmd'], errors='coerce')

    # Filter out rows where 'dcbYmd' is before 2020-01-01, keeping NaT (empty values)
    df = df[(df['dcbYmd'].isna()) | (df['dcbYmd'] >= '2020-01-01')]
    df = df.reset_index(drop=True)
    return df


def index_sungsoo_paeup(df):
    new_df = df[df['siteWhlAddr'].str.contains('성수') | df['rdnWhlAddr'].str.contains('성수')]
    new_df = new_df.reset_index(drop=True)
    return new_df


def integrate_paeup(df):
    """
        성동구 전체 업장 정보 추출
    """
    tqdm.pandas()

    df = df.loc[df['siteWhlAddr'].str.contains('서울특별시', na=False)]

    df['state_in_list'] = df['dtlStateNm'].isin(['폐업', '직권폐업', '말소', '등록취소', '직권말소', '허가취소', '폐업처리', '직권취소', '폐쇄', '지정취소'])
    df['explicit_state_in_list'] = df['dtlStateNm'].isin(['폐업'])

    df['second_word'] = df['siteWhlAddr'].progress_apply(
        lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) >= 3 else None
    )
    df['third_word'] = df['siteWhlAddr'].progress_apply(
        lambda x: x.split()[2] if isinstance(x, str) and len(x.split()) >= 3 else None
    )

    df['dcbYmd'] = pd.to_datetime(df['dcbYmd'], errors='coerce')
    df['dcbYear'] = df['dcbYmd'].dt.year

    #print(entire_df)

    grouped_df = df.groupby(['opnSfTeamCode', 'second_word', 'third_word']).progress_apply(
        lambda group: pd.Series({
            'total_count': group['state_in_list'].size,

            'total_paeup_count': ((group['state_in_list'] == True) & (group['dcbYear'].isin(range(2020, 2025)))).sum(),
            'explicit_paeup_count': ((group['explicit_state_in_list'] == True) & (group['dcbYear'].isin(range(2020, 2025)))).sum(),

            'paeup_2020': ((group['state_in_list'] == True) & (group['dcbYear'] == 2020)).sum(),
            'paeup_2021': ((group['state_in_list'] == True) & (group['dcbYear'] == 2021)).sum(),
            'paeup_2022': ((group['state_in_list'] == True) & (group['dcbYear'] == 2022)).sum(),
            'paeup_2023': ((group['state_in_list'] == True) & (group['dcbYear'] == 2023)).sum(),
            'paeup_2024': ((group['state_in_list'] == True) & (group['dcbYear'] == 2024)).sum()
        })
    ).reset_index()

    grouped_df['total_paeup_years'] = grouped_df[['paeup_2020', 'paeup_2021', 'paeup_2022', 'paeup_2023', 'paeup_2024']].sum(axis=1)

    grouped_df = grouped_df.sort_values(by='total_count', ascending=False)

    return grouped_df


def extract_word_before(match):
    """
        성수동 주소 추출 (정규식 활용)
    """

    pattern = r'([가-힣]+)(?=\d*동|\d+|가|$)'
    result = re.search(pattern, match)
    extracted_word = result.group(1).strip() if result else None
    
    if extracted_word and extracted_word.endswith('동'):
        extracted_word = extracted_word[:-1]
    
    return extracted_word

def calculate_paeup_ratio(group_df):
    group_df = group_df[~group_df['second_word'].str.contains(r'\*')]
    group_df = group_df[~group_df['second_word'].str.contains(r'\d', na=False)]

    group_df = group_df[~group_df['third_word'].str.contains(r'\*')]

    group_df['third_word2'] = group_df['third_word'].apply(extract_word_before)


    group_df['third_word'] = group_df['third_word'].apply(extract_word_before)


    group_df['third_word'] = group_df['third_word'].fillna('')

    group_df = group_df[~group_df['third_word'].str.isnumeric()]


    columns = ['paeup_2020', 'paeup_2021', 'paeup_2022', 'paeup_2023', 'paeup_2024','total_paeup_count', 'explicit_paeup_count', 'total_count']
    group_df = group_df.groupby(['second_word', 'third_word'])[columns].sum().reset_index()


    total_count_mean = group_df['total_count'].mean()

    group_df = group_df[group_df['total_count'] > total_count_mean]


    group_df['paeup_ratio'] = group_df['total_paeup_count'] / group_df['total_count'].replace(0, pd.NA)
    group_df['explicit_paeup_ratio'] = group_df['explicit_paeup_count'] / group_df['total_count'].replace(0, pd.NA)

    return group_df


if __name__ == "__main__":
    entire_paeup_df = make_csv_file(r'6110000_XML.zip')
    entire_paeup_df = filter_cols_under_2020(entire_paeup_df)

    sungsoo_df = index_sungsoo_paeup(entire_paeup_df)

    entire_grouped_df = integrate_paeup(entire_paeup_df)
    paeup_ratio_df = calculate_paeup_ratio(entire_grouped_df)


    sungsoo_df.to_csv(r'store_status_sungsoo_20250115.csv', index=False)
    paeup_ratio_df.to_csv(r'paeup_ratio_added_20250115.csv', index=False)


    #print(paeup_ratio_df.loc[paeup_ratio_df['third_word'].str.startswith('성수')].T)

