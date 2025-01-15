import pandas as pd
from shapely.geometry import Point
from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import ast

# Enable progress_apply
tqdm.pandas()

def fill_공지지가_group(df):
    """
        공시지가 컬럼 전처리
    """

    def fill_공지지가(row):
        if pd.isna(row['공지지가']):
            if '202301' <= row['yyyymm'] <= '202306':
                return df[df['yyyymm'] == '202301']['공지지가'].values[0]
            elif '202307' <= row['yyyymm'] <= '202312':
                return df[df['yyyymm'] == '202307']['공지지가'].values[0]
            elif '202401' <= row['yyyymm'] <= '202405':
                return df[df['yyyymm'].isin(['202301', '202307', '202201', '202207'])]['공지지가'].mean()
        return row['공지지가']
    
    df['공지지가'] = df.apply(fill_공지지가, axis=1)
    return df

def save_long_df():
    """
        일별, 격자별 건축 데이터셋
    """
    df = pd.read_csv('datasets/polygon/shp_to_csv45.csv' , encoding = 'utf-8', header=[0, 1])

    df = df.drop(index = 0)

    df.set_index(('key', 'date'), inplace=True)
    df.index.name = 'gid'

    melted_dfs = []

    for first_level in df.columns.get_level_values(0).unique():
        df_subset = df[first_level]
        
        melted_df = df_subset.melt(
            ignore_index=False,
            var_name='yyyymm',
            value_name=first_level
        ).reset_index()
        
        melted_dfs.append(melted_df)

    df_long = melted_dfs[0]
    for additional_df in melted_dfs[1:]:
        df_long = df_long.merge(additional_df, on=['gid', 'yyyymm'], how='outer')

    df_long = df_long.groupby('gid', group_keys=False).apply(fill_공지지가_group)

    df_long = df_long[(df_long['yyyymm'] >= '202301') & (df_long['yyyymm'] <= '202405')]
    df_long.reset_index(drop=True, inplace=True)

    df_long.to_csv('datasets/polygon/shp_to_csv45_updated.csv' , encoding = 'utf-8')

def find_polygon_mapping(x, y, polygon_df):
    """
        폴리곤 컬럼 전처리
    """
    point = Point(x, y)
    for _, polygon_row in polygon_df.iterrows():
        polygon = wkt.loads(polygon_row['2'])
        if polygon.contains(point):
            return polygon_row['0'], polygon_row['1']
    return None, None

def process_dataframes(dfs, polygon_df, output_paths):
    """
        폴리곤 컬럼 전처리
    """
    for df, output_path in zip(dfs, output_paths):
        if 'x' in df.columns and 'y' in df.columns:
            df[['cell_id_1', 'cell_id_2']] = df.progress_apply(
                lambda row: find_polygon_mapping(row['x'], row['y'], polygon_df), axis=1, result_type='expand'
            )
            df.dropna(inplace=True)
            df.to_csv(output_path, encoding='utf-8')


def calculate_all_station_distances(gdf_polygons, gdf_transports, transport_nm):
    """
        격자 중심과 각 정류장 거리 계산
    """
    centroids = gdf_polygons.geometry.centroid
    
    distance_dicts = [
        {transport_nm: centroid.distance(transport) for transport_nm, transport in zip(gdf_transports[transport_nm], gdf_transports.geometry)}
        for centroid in centroids
    ]
    return distance_dicts

def create_geodataframe(df, x_col, y_col, crs="epsg:5179"):
    """
        x, y 위치 정보를  epsg:5179 체계로 변환
    """
    df['geometry'] = df.apply(lambda row: Point(row[x_col], row[y_col]), axis=1)
    return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)


def combine_grid_transport():
    """
        성수동 45개 격자 위치 정보 따릉이, 버스, 지하철 위치 정보 merge
    """

    polygon_df = pd.read_csv('datasets/polygon/sungsoo_2nd_drop_polygons_45.csv', encoding='utf-8')
    bike_df = pd.read_csv('datasets/bike/seongsu/bike_master_modified.csv', encoding='utf-8')
    bus_df = pd.read_csv('datasets/bus/seongsu/bus_master_modified.csv', encoding='utf-8')
    subway_df = pd.read_csv('datasets/station/seongsu/seongsu_station_master_modified.csv', encoding='utf-8', index_col='index')

    bus_ARS_ID_mapping = bus_df.groupby('cell_id_1')['ARS_ID'].apply(list).to_dict()
    bike_st_id_mapping = bike_df.groupby('cell_id_1')['st_id'].apply(list).to_dict()
    subway_st_nm_mapping = subway_df.groupby('cell_id_1')['st_nm'].apply(list).to_dict()

    polygon_df['bus_ARS_ID_list'] = polygon_df['0'].map(bus_ARS_ID_mapping)
    polygon_df['bike_st_id_list'] = polygon_df['0'].map(bike_st_id_mapping)
    polygon_df['subway_st_nm_list'] = polygon_df['0'].map(subway_st_nm_mapping)

    polygon_df['geometry'] = gpd.GeoSeries.from_wkt(polygon_df['2'])
    gdf_polygons = gpd.GeoDataFrame(polygon_df, geometry='geometry', crs="epsg:5179")  # polygon 위치 정보를 epsg:5179 체계로 변환

    # 교통수단 위치 정보를 epsg:5179 체계로 변환
    gdf_bikes = create_geodataframe(bike_df, 'x', 'y')
    gdf_buses = create_geodataframe(bus_df, 'x', 'y')
    gdf_subways = create_geodataframe(subway_df, 'x', 'y')

    # 격자 중심과 각 정류장의 거리(m) 계산
    gdf_polygons['bike_distances'] = calculate_all_station_distances(gdf_polygons, gdf_bikes, 'st_id')
    gdf_polygons['bus_distances'] = calculate_all_station_distances(gdf_polygons, gdf_buses, 'ARS_ID')
    gdf_polygons['subway_distances'] = calculate_all_station_distances(gdf_polygons, gdf_subways, 'st_nm')
    
    return gdf_polygons

def expand_monthly_to_daily(df, month_col):
    daily_rows = []

    for index, row in df.iterrows():
        month_period = row[month_col]
        year = month_period.year 
        month_num = month_period.month
        
        start_date = pd.Timestamp(year, month_num, 1)
        end_date = start_date + pd.offsets.MonthEnd(0)
        
        days_in_month = pd.date_range(start=start_date, end=end_date, freq='D')

        for day in days_in_month:
            daily_row = {
                'polygon_id1': row['polygon_id1'],
                'polygon_id2': row['polygon_id2'],
                '2': row['2'],
                'bus_ARS_ID_list': row['bus_ARS_ID_list'],
                'bike_st_id_list': row['bike_st_id_list'],
                'subway_st_nm_list': row['subway_st_nm_list'],
                'bike_distances': row['bike_distances'],
                'bus_distances': row['bus_distances'],
                'subway_distances': row['subway_distances'],
                'month': day,
                '건물높이': row['건물높이'],
                '건축면적': row['건축면적'],
                '건폐율': row['건폐율'],
                '공지지가': row['공지지가'],
                '사용승인일': row['사용승인일'],
                '연면적': row['연면적'],
                '용적율': row['용적율']
            }
            daily_rows.append(daily_row)

    daily_df = pd.DataFrame(daily_rows)

    return daily_df

def get_unique_ids(grouped_row):
    """ 
        정류장 id 추출
    """
    unique_ids = set(id for sublist in grouped_row for id in sublist)
    return list(unique_ids)

def parse_station_ids(st_ids):
    """
        정류장 id를 리스트 형태로 반환
    """    

    if pd.isna(st_ids):
        return []  # Return an empty list for NaN
    try:
        return ast.literal_eval(st_ids)  # Convert to list
    except (ValueError, SyntaxError):
        return []  # Return an empty list for malformed strings

def make_polygon_bus_card_df():
    """
        버스 승하차 정보 추출
    """
    polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding='cp949')
    bus_card_df = pd.read_csv('datasets/bus/seongsu/seoungsu_card_bus_data.csv', encoding='utf-8')


    polygon_df['bus_ARS_ID_list'] = polygon_df['bus_ARS_ID_list'].apply(parse_station_ids)
    polygon_df = polygon_df.groupby('polygon_id1')['bus_ARS_ID_list'].agg(list).reset_index()


    polygon_df['bus_ARS_ID_list'] = polygon_df['bus_ARS_ID_list'].apply(get_unique_ids)

    print(polygon_df)


    result_df = pd.DataFrame()


    for index, row in tqdm(polygon_df.iterrows(), total=polygon_df.shape[0], desc="Processing Polygons"):
        bus_ids = row['bus_ARS_ID_list']
        
        if isinstance(bus_ids, list) and bus_ids:
            filtered_buses = bus_card_df[bus_card_df['ARS_ID'].isin(bus_ids)]
            
            filtered_buses['polygon_id'] = row['polygon_id1']
            result_df = pd.concat([result_df, filtered_buses], ignore_index=True)

    result_df_grouped = result_df.groupby(['polygon_id', 'use_date']).agg({
        'board_cnt': ['mean', 'sum'],
        'disembark_cnt': ['mean', 'sum']
    }).reset_index()

    result_df_grouped.columns = ['polygon_id', 'use_date', 
                                'bus_board_cnt_mean', 'bus_board_cnt_sum', 
                                'bus_disembark_cnt_mean', 'bus_disembark_cnt_sum']


    return result_df_grouped


def make_polygon_bike_card_df():
    """
        따릉이 대여, 반납 정보 추출
    """
    polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding='cp949')
    bike_2023 = pd.read_csv('datasets/bike/seongsu/seongsu_combined_bike_202301_202312.csv', encoding='utf-8')
    bike_2024 = pd.read_csv('datasets/bike/seongsu/seongsu_combined_bike_202401_202406.csv', encoding='utf-8')

    bike_df = pd.concat([bike_2023, bike_2024], ignore_index=True)


    bike_df['return_hour'] =  (bike_df['return_hour'].fillna(0) // 100).astype(int)
    bike_df = bike_df.loc[bike_df['return_hour'].isin(range(0, 24))]

    polygon_df['bike_st_id_list'] = polygon_df['bike_st_id_list'].apply(parse_station_ids)

    polygon_df = polygon_df.groupby('polygon_id1')['bike_st_id_list'].agg(list).reset_index()
    polygon_df['bike_st_id_list'] = polygon_df['bike_st_id_list'].apply(get_unique_ids)


    result_df = pd.DataFrame()


    for index, row in polygon_df.iterrows():
        bus_ids = row['bike_st_id_list']
        if isinstance(bus_ids, list) and bus_ids:
            filtered_bikes = bike_df[bike_df['return_st_id'].isin(bus_ids)]
            
            filtered_bikes['polygon_id'] = row['polygon_id1']
            
            result_df = pd.concat([result_df, filtered_bikes], ignore_index=True)

    result_df_grouped = result_df.groupby(['polygon_id', 'date', 'return_hour']).agg({
        'trip_count': ['mean'],
        'trip_minute': ['mean']
    }).reset_index()

    result_df_grouped = result_df_grouped.rename(columns={
        'trip_count': 'bike_trip_count',
        'trip_minute': 'bike_trip_minute'
    })

    pivot_df = result_df_grouped.pivot(index=['polygon_id', 'date'], columns='return_hour', values=['bike_trip_count', 'bike_trip_minute'])
    pivot_df.columns = [f"{metric}_hour_{hour}" for metric, hour in pivot_df.columns]

    pivot_df.reset_index(inplace=True)
    pivot_df.fillna(0, inplace = True)

    return pivot_df

#make_polygon_bike_card_df()

def make_polygon_subway_card_df():
    """
        지하철 승하차 정보 추출
    """

    polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding='cp949')
    subway_df = pd.read_csv('datasets/station/seongsu/seongsu_card_subway_data.csv', encoding='utf-8')

    polygon_df['subway_st_nm_list'] = polygon_df['subway_st_nm_list'].apply(parse_station_ids)

    polygon_df = polygon_df.groupby('polygon_id1')['subway_st_nm_list'].agg(list).reset_index()
    polygon_df['subway_st_nm_list'] = polygon_df['subway_st_nm_list'].apply(get_unique_ids)

    result_df = pd.DataFrame()

    for index, row in polygon_df.iterrows():
        bus_ids = row['subway_st_nm_list']
        if isinstance(bus_ids, list) and bus_ids:
            filtered_bikes = subway_df[subway_df['st_nm'].isin(bus_ids)]
            filtered_bikes['polygon_id'] = row['polygon_id1']
            result_df = pd.concat([result_df, filtered_bikes], ignore_index=True)

    result_df_grouped = result_df.groupby(['polygon_id', 'use_date']).agg({
        'board_cnt': ['mean'],
        'disembark_cnt': ['mean']
    }).reset_index()

    result_df_grouped.columns = ['polygon_id', 'use_date', 
                                'subway_board_cnt_mean', 'subway_disembark_cnt_mean']

    return result_df_grouped


def make_board():
    """
        교통 수단 승하차 정보 추출
    """

    bus_card_df = make_polygon_bus_card_df()
    bike_card_df = make_polygon_bike_card_df()
    subway_card_df = make_polygon_subway_card_df()

    bike_card_df = bike_card_df.rename(columns={'date': 'use_date'})
    merged_df = pd.merge(bus_card_df, bike_card_df, on=['polygon_id', 'use_date'], how='outer')
    merged_df = pd.merge(merged_df, subway_card_df, on=['polygon_id', 'use_date'], how='outer')

    merged_df.fillna(0, inplace =True)

    
    merged_df['use_date'] = pd.to_datetime(merged_df['use_date'].astype(str), format='%Y%m%d')
    merged_df = merged_df.loc[merged_df['use_date'] < '2024-06-01'] 
    merged_df.to_csv('datasets/polygon/polygons_45_board.csv', encoding = 'cp949')


def make_transport_building_board():
    """
        교통수단 & 건물 & 승하차 정보 merge
    """
    board_df = pd.read_csv('datasets/polygon/polygons_45_board.csv', encoding = 'cp949')
    polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding = 'cp949')

    polygon_df = polygon_df.loc[polygon_df['month'] < '2024-05-01'] 

    polygon_df_board_df = pd.merge(polygon_df, board_df, left_on = ['polygon_id1', 'month'], right_on = ['polygon_id', 'use_date'], how = 'left')

    polygon_df_board_df.fillna(0, inplace = True)
    polygon_df_board_df.drop(axis = 1, columns = ['polygon_id', 'Unnamed: 0_y', 'use_date'], inplace =True)
    polygon_df_board_df.rename(columns = {'month' : 'date'}, inplace = True)

    polygon_df_board_df.to_csv('datasets/polygon/polygons_45_building_transport_board.csv', encoding = 'cp949')


def make_transport_building_board_weather():
    """
        교통수단 & 빌딩 & 승하차 & 날씨 정보 merge
    """

    df = pd.read_csv('datasets/polygon/polygons_45_building_transport_board.csv', encoding = 'cp949')
    weather_df = pd.read_excel('datasets/weather/weather.xlsx')
    df['date'] = pd.to_datetime(df['date'])
    weather_df['일시'] = pd.to_datetime(weather_df['일시'])
    merged_df = pd.merge(df, weather_df, left_on='date', right_on = '일시',how='left')

    merged_df.fillna(0, inplace = True)
    merged_df.to_csv('datasets/polygon/building_transport_board_weather.csv', encoding = 'cp949')



def get_sales():
    """
        업종이 필터링된 매출 정보 데이터 셋
    """

    sales_df = pd.read_csv('datasets/sales/card_category_conditioned.csv', encoding = 'utf-8')
    polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding = 'cp949')

    sales_df.loc[:,'date'] = pd.to_datetime(sales_df['date'])
    sales_df.loc[:,'cate_mask']= 5

    sales_df.loc[sales_df['category']=='한식', 'cate_mask'] = 0
    sales_df.loc[sales_df['category'].isin(['중식','양식','일식']), 'cate_mask'] = 1
    sales_df.loc[sales_df['category'].isin(['패스트푸드','커피전문점','제과점']), 'cate_mask'] = 2


    sales_df = sales_df.loc[sales_df.cate_mask!=5].reset_index(drop=True)

    sales_df = (sales_df.loc[sales_df['code250'].isin(polygon_df['polygon_id1'].unique())]).reset_index(drop=True)

    sales_df.loc[:, 'sales_day'] = pd.to_datetime(sales_df['date']).dt.day

    sales_df['date'] = pd.to_datetime(sales_df['date'])

    sales_df.drop(axis = 1, columns = ['category', 'before_group_count'], inplace =True)

    sales_df.to_csv('datasets/sales/card_category_conditioned_processed.csv', encoding = 'cp949')

    return sales_df

def make_transport_building():
    transport_df = combine_grid_transport() # 교통수단 데이터
    building_df = pd.read_csv('datasets/polygon/shp_to_csv45_updated.csv', encoding='utf-8') # 건물 데이터
    merged_df = transport_df.merge(building_df, left_on='0', right_on='gid', how='inner') # 교통수단 & 건물 데이터

    # 필요없는 컬럼 지우기 및 변수 이름 변환
    merged_df.drop(axis = 1, columns = ['Unnamed: 0_x', 'Unnamed: 0_y','gid'], inplace = True)
    merged_df['yyyymm'] = pd.to_datetime(merged_df['yyyymm'], format='%Y%m')
    merged_df['yyyymm'] = merged_df['yyyymm'].dt.to_period('M')
    merged_df.rename(columns = {'0': 'polygon_id1', '1': 'polygon_id2', 'yyyymm' : 'month'}, inplace =True)
    merged_df.reset_index(drop=True, inplace=True)

    # month 변수 형변환
    daily_df = expand_monthly_to_daily(merged_df, 'month')

    # 최종 교통수단 & 건물 데이터
    daily_df.to_csv('datasets/polygon/polygons_45_building_transport.csv', encoding='cp949')

def main():
    save_long_df() #건물데이터 추출
    make_transport_building() # 교통수단 & 건물 데이터 merge
    make_board() # 승하차 데이터 추출
    make_transport_building_board() # 교통수단 & 건물 & 승하차 데이터 merge
    make_transport_building_board_weather() # 교통수단 & 건물 & 승하차 & 날씨 데이터 merge
    get_sales() # 매출 데이터 추출

if __name__ == "__main__":
    main()
