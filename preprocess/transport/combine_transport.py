# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:29:37 2025

@author: dohyeon
"""

import pandas as pd
from shapely.geometry import Point
from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import ast



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
    new_df = df.copy(deep=True)
    new_df['geometry'] = new_df.apply(lambda row: Point(row[x_col], row[y_col]), axis=1)
    return gpd.GeoDataFrame(new_df, geometry='geometry', crs=crs)


def combine_grid_transport(polygon_df, bike_df, bus_df, subway_df):
    """
        성수동 45개 격자 위치 정보 따릉이, 버스, 지하철 위치 정보 merge
    """

    polygon_df1 = polygon_df.copy(deep=True)
    #polygon_df = pd.read_csv('datasets/polygon/sungsoo_2nd_drop_polygons_45.csv', encoding='utf-8')
    #bike_df = pd.read_csv('datasets/bike/seongsu/bike_master_modified.csv', encoding='utf-8')
    #bus_df = pd.read_csv('datasets/bus/seongsu/bus_master_modified.csv', encoding='utf-8')
    #subway_df = pd.read_csv('datasets/station/seongsu/seongsu_station_master_modified.csv', encoding='utf-8', index_col='index')

    bus_ARS_ID_mapping = bus_df.groupby('cell_id_1')['ARS_ID'].apply(list).to_dict()
    bike_st_id_mapping = bike_df.groupby('cell_id_1')['st_id'].apply(list).to_dict()
    subway_st_nm_mapping = subway_df.groupby('cell_id_1')['st_nm'].apply(list).to_dict()

    polygon_df1['bus_ARS_ID_list'] = polygon_df1['0'].map(bus_ARS_ID_mapping)
    polygon_df1['bike_st_id_list'] = polygon_df1['0'].map(bike_st_id_mapping)
    polygon_df1['subway_st_nm_list'] = polygon_df1['0'].map(subway_st_nm_mapping)

    polygon_df1['geometry'] = gpd.GeoSeries.from_wkt(polygon_df1['2'])
    gdf_polygons = gpd.GeoDataFrame(polygon_df1, geometry='geometry', crs="epsg:5179")  # polygon 위치 정보를 epsg:5179 체계로 변환

    # 교통수단 위치 정보를 epsg:5179 체계로 변환
    gdf_bikes = create_geodataframe(bike_df, 'x', 'y')
    gdf_buses = create_geodataframe(bus_df, 'x', 'y')
    gdf_subways = create_geodataframe(subway_df, 'x', 'y')

    # 격자 중심과 각 정류장의 거리(m) 계산
    gdf_polygons['bike_distances'] = calculate_all_station_distances(gdf_polygons, gdf_bikes, 'st_id')
    gdf_polygons['bus_distances'] = calculate_all_station_distances(gdf_polygons, gdf_buses, 'ARS_ID')
    gdf_polygons['subway_distances'] = calculate_all_station_distances(gdf_polygons, gdf_subways, 'st_nm')
    
    return gdf_polygons

###

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


def make_polygon_subway_card_df1(polygon_df, subway_usage_df):
    """
        지하철 승하차 정보 추출
    """

    #polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding='cp949')
    #subway_df = pd.read_csv('datasets/station/seongsu/seongsu_card_subway_data.csv', encoding='utf-8')

    polygon_df['subway_st_nm_list'] = polygon_df['subway_st_nm_list'].apply(parse_station_ids)

    polygon_df = polygon_df.groupby('polygon_id1')['subway_st_nm_list'].agg(list).reset_index()
    polygon_df['subway_st_nm_list'] = polygon_df['subway_st_nm_list'].apply(get_unique_ids)

    result_df = pd.DataFrame()

    for index, row in polygon_df.iterrows():
        bus_ids = row['subway_st_nm_list']
        if isinstance(bus_ids, list) and bus_ids:
            filtered_bikes = subway_usage_df[subway_usage_df['st_nm'].isin(bus_ids)]
            filtered_bikes['polygon_id'] = row['polygon_id1']
            result_df = pd.concat([result_df, filtered_bikes], ignore_index=True)

    result_df_grouped = result_df.groupby(['polygon_id', 'use_date']).agg({
        'board_cnt': ['mean'],
        'disembark_cnt': ['mean']
    }).reset_index()

    result_df_grouped.columns = ['polygon_id', 'use_date', 
                                'subway_board_cnt_mean', 'subway_disembark_cnt_mean']

    return result_df_grouped

def make_polygon_subway_card_df(polygon_df, subway_usage_df):
    """
        지하철 승하차 정보 추출
    """

    #polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding='cp949')
    #subway_df = pd.read_csv('datasets/station/seongsu/seongsu_card_subway_data.csv', encoding='utf-8')

    #polygon_df['subway_st_nm_list'] = polygon_df['subway_st_nm_list'].apply(parse_station_ids)

    #polygon_df = polygon_df.groupby('polygon_id1')['subway_st_nm_list'].agg(list).reset_index()
    #polygon_df['subway_st_nm_list'] = polygon_df['subway_st_nm_list'].apply(get_unique_ids)

    result_df = pd.DataFrame()

    for index, row in polygon_df.iterrows():
        bus_ids = row['subway_st_nm_list']
        if isinstance(bus_ids, list) and bus_ids:
            filtered_bikes = subway_usage_df[subway_usage_df['st_nm'].isin(bus_ids)]
            filtered_bikes['polygon_id'] = row['0']
            result_df = pd.concat([result_df, filtered_bikes], ignore_index=True)

    result_df_grouped = result_df.groupby(['polygon_id', 'use_date']).agg({
        'board_cnt': ['mean'],
        'disembark_cnt': ['mean']
    }).reset_index()

    result_df_grouped.columns = ['polygon_id', 'use_date', 
                                'subway_board_cnt_mean', 'subway_disembark_cnt_mean']

    return result_df_grouped


def make_polygon_bus_card_df(polygon_df, bus_usage_df):
    """
        버스 승하차 정보 추출
    """
    #polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding='cp949')
    #bus_card_df = pd.read_csv('datasets/bus/seongsu/seoungsu_card_bus_data.csv', encoding='utf-8')


    #polygon_df['bus_ARS_ID_list'] = polygon_df['bus_ARS_ID_list'].apply(parse_station_ids)
    #polygon_df = polygon_df.groupby('polygon_id1')['bus_ARS_ID_list'].agg(list).reset_index()


    #polygon_df['bus_ARS_ID_list'] = polygon_df['bus_ARS_ID_list'].apply(get_unique_ids)

    #print(polygon_df)


    result_df = pd.DataFrame()


    for index, row in tqdm(polygon_df.iterrows(), total=polygon_df.shape[0], desc="Processing Polygons"):
        bus_ids = row['bus_ARS_ID_list']
        
        if isinstance(bus_ids, list) and bus_ids:
            filtered_buses = bus_usage_df[bus_usage_df['ARS_ID'].isin(bus_ids)]
            
            filtered_buses['polygon_id'] = row['0']
            result_df = pd.concat([result_df, filtered_buses], ignore_index=True)

    result_df_grouped = result_df.groupby(['polygon_id', 'use_date']).agg({
        'board_cnt': ['mean', 'sum'],
        'disembark_cnt': ['mean', 'sum']
    }).reset_index()

    result_df_grouped.columns = ['polygon_id', 'use_date', 
                                'bus_board_cnt_mean', 'bus_board_cnt_sum', 
                                'bus_disembark_cnt_mean', 'bus_disembark_cnt_sum']


    return result_df_grouped

def make_polygon_bike_card_df(polygon_df, bike_usage_2023, bike_usage_2024):
    """
        따릉이 대여, 반납 정보 추출
    """
    #polygon_df = pd.read_csv('datasets/polygon/polygons_45_building_transport.csv', encoding='cp949')
    #bike_2023 = pd.read_csv('datasets/bike/seongsu/seongsu_combined_bike_202301_202312.csv', encoding='utf-8')
    #bike_2024 = pd.read_csv('datasets/bike/seongsu/seongsu_combined_bike_202401_202406.csv', encoding='utf-8')

    #bike_df = pd.concat([bike_2023, bike_2024], ignore_index=True)
    bike_df = pd.concat([bike_usage_2023, bike_usage_2024], ignore_index=True)


    bike_df['return_hour'] =  (bike_df['return_hour'].fillna(0) // 100).astype(int)
    bike_df = bike_df.loc[bike_df['return_hour'].isin(range(0, 24))]

    #polygon_df['bike_st_id_list'] = polygon_df['bike_st_id_list'].apply(parse_station_ids)

    #polygon_df = polygon_df.groupby('polygon_id1')['bike_st_id_list'].agg(list).reset_index()
    #polygon_df['bike_st_id_list'] = polygon_df['bike_st_id_list'].apply(get_unique_ids)


    result_df = pd.DataFrame()


    for index, row in polygon_df.iterrows():
        bus_ids = row['bike_st_id_list']
        if isinstance(bus_ids, list) and bus_ids:
            filtered_bikes = bike_df[bike_df['return_st_id'].isin(bus_ids)]
            
            filtered_bikes['polygon_id'] = row['0']
            
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


#%%
if __name__ == "__main__":
    polygon_df = pd.read_csv(r'..\sales\sungsoo_2nd_drop_polygons_45.csv', index_col=0)
    bike_df = pd.read_csv(r'bike\bike_master_modified.csv', index_col=0)
    bus_df = pd.read_csv(r'bus\bus_master_modified.csv', index_col=0)
    subway_df = pd.read_csv(r'station\seongsu_station_master_modified.csv', index_col=0)
    transport_df = combine_grid_transport(polygon_df, bike_df, bus_df, subway_df)

    transport_df

    transport_df.to_csv('combined_transport_distance.csv', index=False)







#%%


