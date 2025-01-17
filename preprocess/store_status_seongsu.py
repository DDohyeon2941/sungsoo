import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_xml_row(row, common_columns):
    """
        xml 파일 row 가져오기
        성동구 코드 opnSfTeamCode - 303000
    """
    row_data = {col: '' for col in common_columns}
    for child in row:
        if child.tag in common_columns:
            row_data[child.tag] = child.text.strip() if child.text else ''
    
    opnSfTeamCode = row_data.get('opnSfTeamCode', '')
    if '3030000' not in opnSfTeamCode:
        return None


    return row_data

def make_csv_file():
    """
        XML 파일 -> csv 파일 변환
    """

    zip_file_path = './datasets/6110000_XML.zip'

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


    csv_file_path = './datasets/6110000_entire.csv'
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    print(f"Saved {csv_file_path}")

def seoungsu_csv_file():
    """
        성수 업장 정보 추출
    """
    df = pd.read_csv('./datasets/6110000_entire.csv', encoding='utf-8-sig')

    to_remove_columns = ['opnSvcId', 'mgtNo', 'apvCancelYmd', 'clgStdt', 'clgEnddt', 'ropnYmd', 'siteTel'
                        , 'siteArea', 'lastModTs', 'updateGbn', 'updateDt']

    df = df.drop(to_remove_columns ,axis = 1)

    df['dcbYmd'] = pd.to_datetime(df['dcbYmd'], errors='coerce')

    df = df[(df['dcbYmd'].isna()) | (df['dcbYmd'] >= '2020-01-01')]


    df = df[df['siteWhlAddr'].str.contains('성수') | df['rdnWhlAddr'].str.contains('성수')]
    df = df.reset_index(drop=True)

    df.to_csv('./datasets/store_status_seungsu.csv', index=False, encoding='utf-8-sig')


def entire_csv_file():
    """
        성동구 전체 업장 정보 추출
    """
    df = pd.read_csv('./datasets/6110000_entire.csv', encoding='utf-8-sig')

    to_remove_columns = ['opnSvcId', 'mgtNo', 'apvCancelYmd', 'clgStdt', 'clgEnddt', 'ropnYmd', 'siteTel'
                        , 'siteArea', 'lastModTs', 'updateGbn', 'updateDt']

    df = df.drop(to_remove_columns ,axis = 1)

    df['dcbYmd'] = pd.to_datetime(df['dcbYmd'], errors='coerce')

    # Filter out rows where 'dcbYmd' is before 2020-01-01, keeping NaT (empty values)
    df = df[(df['dcbYmd'].isna()) | (df['dcbYmd'] >= '2020-01-01')]

    df = df.reset_index(drop=True)

    df.to_csv('./datasets/6110000_entire_modified.csv', index=False, encoding='utf-8-sig')

def entired_modify():
    """
        성동구 전체 업장 정보 추출
    """
    tqdm.pandas()

    entire_df = pd.read_csv('./datasets/6110000_entire_modified.csv', encoding='utf-8-sig', low_memory=False)

    entire_df = entire_df.loc[entire_df['siteWhlAddr'].str.contains('서울특별시', na=False)]

    entire_df['state_in_list'] = entire_df['dtlStateNm'].isin(['폐업', '직권폐업', '말소', '등록취소', '직권말소', '허가취소', '폐업처리', '직권취소', '폐쇄', '지정취소'])
    entire_df['explicit_state_in_list'] = entire_df['dtlStateNm'].isin(['폐업'])

    entire_df['second_word'] = entire_df['siteWhlAddr'].progress_apply(
        lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) >= 3 else None
    )
    entire_df['third_word'] = entire_df['siteWhlAddr'].progress_apply(
        lambda x: x.split()[2] if isinstance(x, str) and len(x.split()) >= 3 else None
    )

    entire_df['dcbYmd'] = pd.to_datetime(entire_df['dcbYmd'], errors='coerce')
    entire_df['dcbYear'] = entire_df['dcbYmd'].dt.year

    #print(entire_df)

    grouped_df = entire_df.groupby(['opnSfTeamCode', 'second_word', 'third_word']).progress_apply(
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

    print(grouped_df.head())

    grouped_df.to_csv('./datasets/seoul_paeup_year_grouped.csv', index=True, encoding='utf-8-sig')

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

def make_paeup_ratio():
    """
        성수동 폐업률 비율 확인
    """
    group_df = pd.read_csv('./datasets/seoul_paeup_year_grouped.csv', encoding='utf-8-sig', low_memory=False)


    group_df = group_df[~group_df['second_word'].str.contains(r'\*')]
    group_df = group_df[~group_df['second_word'].str.contains(r'\d', na=False)]

    group_df = group_df[~group_df['third_word'].str.contains(r'\*')]

    group_df['third_word2'] = group_df['third_word'].apply(extract_word_before)


    group_df['third_word'] = group_df['third_word'].apply(extract_word_before)


    group_df['third_word'] = group_df['third_word'].fillna('')

    group_df = group_df[~group_df['third_word'].str.isnumeric()]


    columns = ['paeup_2020', 'paeup_2021', 'paeup_2022', 'paeup_2023', 'total_paeup_count', 'explicit_paeup_count', 'total_count']
    group_df = group_df.groupby(['second_word', 'third_word'])[columns].sum().reset_index()


    total_count_mean = group_df['total_count'].mean()

    group_df = group_df[group_df['total_count'] > total_count_mean]


    group_df['paeup_ratio'] = group_df['total_paeup_count'] / group_df['total_count'].replace(0, pd.NA)
    group_df['explicit_paeup_ratio'] = group_df['explicit_paeup_count'] / group_df['total_count'].replace(0, pd.NA)

    group_df = group_df.sort_values(by='explicit_paeup_ratio', ascending=True)

    print(group_df.loc[group_df['third_word'].str.startswith('성수')])

    group_df.to_csv('./datasets/paeup_ratio_added.csv', index=True, encoding='utf-8-sig')


def main():
    # 성수동 업장 정보 추출
    make_csv_file()
    seoungsu_csv_file()

    # 성수동 폐업률 비율 추출
    entire_csv_file()
    entired_modify()
    make_paeup_ratio()

if __name__ == "__main__":
    main()