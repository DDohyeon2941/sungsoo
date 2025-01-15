import pandas as pd
import os
from pyproj import Transformer
from shapely.geometry import Point

def extract_subway_info():

    master_df = pd.read_csv('./datasets/station/raw/station_master.csv', encoding = 'CP949')

    new_column_names_master = ['row', 'line', 'unique_code', 'st_nm', 'latitude', 'longtitude', 'est_date']

    master_df.columns = new_column_names_master

    dataframes = []

    directory = './datasets/station/raw/'
    encodings_to_try = ['utf-8-sig', 'cp949', 'utf-16', 'latin1']
    for filename in os.listdir(directory):
        if filename.startswith('CARD_SUBWAY_MONTH_'):
            file_path = os.path.join(directory, filename)
            
            for encoding in encodings_to_try:
                try:
                    card_df = pd.read_csv(f'{file_path}', encoding=encoding)
                    print(f"Successfully read {filename} using {encoding} encoding.")

                    break
                except (UnicodeDecodeError, FileNotFoundError) as e:
                    print(f"Could not decode {filename} using {encoding} encoding: {e}")

            new_column_names_card = ['use_date', 'line', 'st_nm', 'board_cnt', 'disembark_cnt', 'reg_date']

            card_df.columns = new_column_names_card

            card_df = card_df.loc[card_df['st_nm'].isin(['성수', '뚝섬', '서울숲'])].reset_index(drop = True)

            dataframes.append(card_df)

            print(f"Processed file: {filename}")

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv('./datasets/station/seongsu/seoungsu_card_subway_data.csv', index=False, encoding='utf-8-sig')

def main():
    extract_subway_info()

if __name__ == "__main__":
    main()