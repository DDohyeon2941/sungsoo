import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely import wkt
from tqdm import tqdm
import zipfile
import numpy as np

transformer = Transformer.from_crs("epsg:4326", "epsg:5179")

def transform_coordinates(lat, lon):
        x, y = transformer.transform(lat, lon)
        return x, y

def get_seongsu_bike(master_df, polygon_df):
    """
        성수 bike 정류장 정보 가져오기
    """
    new_column_names_master = ['st_id', 'addr1', 'addr2', 'latitude', 'longtitude']
    master_df.columns = new_column_names_master

    master_df['y'], master_df['x'] = zip(*master_df.apply(lambda row: transform_coordinates(row['latitude'], row['longtitude']), axis=1))

    new_column_names_polygon = ['cell_id_1', 'cell_id_2', 'cell_info']
    polygon_df.columns = new_column_names_polygon

    def find_polygon_mapping(x, y):
        point = Point(x, y)
        for _, polygon_row in polygon_df.iterrows():
            polygon = wkt.loads(polygon_row['cell_info'])
            if polygon.contains(point):
                return polygon_row['cell_id_1'], polygon_row['cell_id_2']
        return None, None 

    tqdm.pandas()

    master_df[['cell_id_1', 'cell_id_2']] = master_df.progress_apply(
        lambda row: find_polygon_mapping(row['x'], row['y']), axis=1, result_type='expand'
    )


    master_df = master_df.loc[(master_df['cell_id_1'].notna()) | (master_df['cell_id_2'].notna())]
    master_df = master_df.drop(['cell_id_1', 'cell_id_2'], axis = 1)
    return master_df

def calculate_return_hour(row):
    """
        반납 시간 정보 계산
    """

    rent_hour = row['rent_hour']
    trip_minute = row['trip_minute']

    hour = rent_hour // 100
    minute = rent_hour % 100

    total_minutes = minute + trip_minute

    new_hour = hour + total_minutes // 60
    new_minute = total_minutes % 60

    new_minute = (new_minute // 5) * 5

    if new_minute >= 60:
        new_minute -= 60
        new_hour += 1

    return_hour = new_hour * 100 + new_minute
    return return_hour

def extract_bike_info(master_df, start_month, end_month):
    """
        start_month: yyyymm
        end_month: yyyymm
        따릉이 승하차 정보 추출
    """
    valid_stations = master_df['st_id'].tolist()

    combined_csv_file = 'datasets/bike/seongsu/seongsu_combined_bike_202301_202406.csv'

    all_data = []

    for month in tqdm(range(start_month, end_month), desc='Processing CSV files'):
        zip_file = f'datasets/bike/raw/tpss_bcycl_od_statnhm_{month}.zip'
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                for csv_file in z.namelist():
                    if csv_file.endswith('.csv'):
                        with z.open(csv_file) as f:
                            
                            trip_data = pd.read_csv(f, encoding = 'cp949')

                            if '집계_기준' in trip_data.columns:
                                trip_data = trip_data.loc[trip_data['집계_기준'] == '출발시간'] 

                            trip_data = trip_data.loc[:, ~trip_data.columns.str.contains('^Unnamed|^$', regex=True)]

                            required_columns = ['집계_기준', '시작_대여소명', '종료_대여소명', '']

                            existing_columns = [col for col in required_columns if col in trip_data.columns]

                            trip_data = trip_data.drop(columns=existing_columns)

                            new_column_names_trip = ['date', 'rent_hour', 'rent_st_id', 'return_st_id', 'trip_count', 'trip_minute', 'trip_distance']
                            
                            try:
                                trip_data.columns = new_column_names_trip
                            except ValueError as e:
                                print(f"ValueError: {e}")
                                print('filename: ', f)
                                print('trip_data.columns: ', trip_data.columns)
                                print(f"Expected {len(new_column_names_trip)} elements, but got {trip_data.shape[1]} elements.")
                            

                            trip_data['return_hour'] = trip_data.apply(calculate_return_hour, axis=1)
                        
                            trip_data['seongsu_in'] = np.where(trip_data['return_st_id'].isin(valid_stations), 1, np.nan)
                            
                            filtered_data = trip_data[trip_data['seongsu_in'].notna()]

                            all_data.append(filtered_data)
        except FileNotFoundError:
            print(f"File not found: {zip_file}")
            continue

    combined_data = pd.concat(all_data, ignore_index=True)

    combined_data = combined_data.sort_values(by='date', ascending=True)
    print(f"Combined data saved to {combined_csv_file}")

    return combined_data




if __name__ == "__main__":

    master_df = pd.read_csv('station_master.csv', encoding='cp949')
    polygon_df= pd.read_csv(r'..\..\sales\sungsoo_2nd_drop_polygons_45.csv', index_col=0)

    added_master_df = get_seongsu_bike(master_df, polygon_df)

    final_df = extract_bike_info(added_master_df, '202301', '202406')

    added_master_df.to_csv(r'bike_master.csv', index=False)
    final_df.to_csv(r'seongsu_combined_bike_202301_202406.csv', index=False, encoding = 'cp949')


    """
    final_df 파일은 202301-202406까지의 데이터셋이나, 이를 2023년과 2024년으로 나누어 저장하였음
    하지만 이에 대한 코드는 생략

    출력정보: seongsu_bike_2023.csv, seongsu_bike_2024.csv

    """





