import pandas as pd
import os

def extract_subway_info(master_df, raw_dataset_dir):

    new_column_names_master = ['row', 'line', 'unique_code', 'st_nm', 'latitude', 'longtitude', 'est_date']

    master_df.columns = new_column_names_master

    dataframes = []

    encodings_to_try = ['utf-8-sig', 'cp949', 'utf-16', 'latin1']
    for filename in os.listdir(raw_dataset_dir):
        if filename.startswith('CARD_SUBWAY_MONTH_'):
            file_path = os.path.join(raw_dataset_dir, filename)
            
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
    return combined_df


if __name__ == "__main__":


    master_df = pd.read_csv('station_master.csv', encoding = 'CP949')
    dataset_dir = r'dataset'

    subway_df = extract_subway_info(master_df, dataset_dir)

    subway_df.to_csv(r'seoungsu_card_subway_data.csv', index=False, encoding='utf-8-sig')

















