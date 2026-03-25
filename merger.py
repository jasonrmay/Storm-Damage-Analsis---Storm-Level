import os

import pandas as pd

class Merge():
    def __init__(self, data_dict, year, download=True, save_path=r"./merged_data/"):
        self.data_dict = data_dict
        self.year = year
        self.download = download
        self.save_path = save_path


    def merge(self):



        # first start with storm_data as it is our parent df.
        storm_df = self.data_dict['storm_data']
        del self.data_dict['storm_data']

        # merge the temp anomaly data on the month and the fips
        merged_df = storm_df.merge(self.data_dict['temp_anomaly'], on=['MONTH', 'FIPS'], how='left')
        del self.data_dict['temp_anomaly']

        # merge the roni data on MONTH_NAME
        merged_df = merged_df.merge(self.data_dict['roni'], on=['MONTH_NAME'], how='left')
        del self.data_dict['roni']

        # for the rest of the dataframes, we will merge on FIPS, preserving left.
        for key, df in self.data_dict.items():
            merged_df = merged_df.merge(df, on='FIPS', how='left')
        # empty out the data dict to free up memory
        self.data_dict.clear()

        # fill NA in coastal type columns with 'inland' 
        merged_df['COASTAL_TYPE_SHORELINE'] = merged_df['COASTAL_TYPE_SHORELINE'].fillna('inland')
        merged_df['COASTAL_TYPE_WATERSHED'] = merged_df['COASTAL_TYPE_WATERSHED'].fillna('inland')



        # ----- final cleaning steps -----

        # make population a numeric column
        merged_df['POPULATION'] = pd.to_numeric(merged_df['POPULATION'], errors='coerce')



        if self.download:
            # check if save path exists, if not create it
            if not os.path.exists(self.save_path + str(self.year)):
                os.makedirs(self.save_path + str(self.year))
            merged_df.to_csv(f"{self.save_path + str(self.year)}/merged_data.csv", index=False)

            # check for downloaded file
            if not os.path.exists(f"{self.save_path + str(self.year)}/merged_data.csv"):
                print(f"Error: Merged data for {self.year} did not download.")

        return merged_df
