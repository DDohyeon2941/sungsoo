import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely import wkt
from tqdm import tqdm
import os

transformer = Transformer.from_crs("epsg:4326", "epsg:5179")

def transform_coordinates(lat, lon):
        x, y = transformer.transform(lat, lon)
        return x, y

def get_seongsu_bus(master_df, polygon_df ):
    """
        성수동 버스 정류장 정보 가져오기
    """

    master_df = pd.read_excel('datasets/bus/raw/station_master.xlsx')
    new_column_names_master = ['NODE_ID', 'ARS_ID', 'st_nm', 'longtitude', 'latitude', 'st_tp']
    master_df.columns = new_column_names_master

    master_df['y'], master_df['x'] = zip(*master_df.apply(lambda row: transform_coordinates(row['latitude'], row['longtitude']), axis=1))

    #polygon_df = pd.read_csv('datasets/polygon/sungsoo_match_60_polygon.csv', encoding='ISO-8859-1')
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

    return master_df


def extract_bus_info(master_df, raw_dataset_dir):

    master_df['ARS_ID'] = '0' + master_df['ARS_ID'].astype(str)
    dataframes = []

    encodings_to_try = ['utf-8-sig', 'cp949', 'utf-16', 'latin1']
    for filename in os.listdir(raw_dataset_dir):
        if filename.startswith('BUS_STATION_BOARDING_MONTH_'):
            file_path = os.path.join(raw_dataset_dir, filename)
            
            for encoding in encodings_to_try:
                try:
                    card_df = pd.read_csv(f'{file_path}', encoding=encoding)
                    

                    print(f"Successfully read {filename} using {encoding} encoding.")

                    break
                except (UnicodeDecodeError, FileNotFoundError) as e:
                    print(f"Could not decode {filename} using {encoding} encoding: {e}")

            new_column_names_card = ['use_date', 'line_number', 'line_nm', 'st_id', 'ARS_ID', 'st_nm', 'board_cnt', 'disembark_cnt', 'reg_date']

            card_df.columns = new_column_names_card

            card_df = card_df.groupby(['use_date', 'ARS_ID']).agg({
                'board_cnt': 'sum',
                'disembark_cnt': 'sum'
            }).reset_index()


            card_df = card_df.loc[card_df['ARS_ID'].isin(master_df['ARS_ID'])].reset_index(drop = True)

            print('card_df: ', card_df)

            dataframes.append(card_df)

            print(f"Processed file: {filename}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


if __name__ == "__main__":

    master_df = pd.read_excel('station_master.xlsx')
    polygon_df= pd.read_csv(r'..\..\sales\sungsoo_2nd_drop_polygons_45.csv', index_col=0)

    added_master_df = get_seongsu_bus(master_df, polygon_df )

    dataset_dir = r'raw'

    final_df = extract_bus_info(master_df, dataset_dir)

    added_master_df.to_csv(r'bus_master.csv', index=False)
    final_df.to_csv(r'seoungsu_card_bus_data.csv', index=False, encoding='utf-8-sig')





