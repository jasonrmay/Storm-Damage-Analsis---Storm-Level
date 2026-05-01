import os
from io import StringIO
import pandas as pd
import numpy as np
import requests
import re

class DataGetter:
    """
    A class to download the csv files for our project.

    Parameters
    ----------
    
    """
    def __init__(self, raw_out_dir = "./rawData", CENSUS_KEY = None, year = 2023, download = False,
                  roni_raw_data_path = r".\static_data\Coastal_data_2010\coastal_counties_2010.csv",
                  coastal_type_raw_data_path = r".\static_data\Coastal_data_2010\coastal_type_data_2010.csv"):
        self.directory = raw_out_dir
        self.api_key = CENSUS_KEY
        self.year = year
        self.download = download
        self.roni_raw_data_path = roni_raw_data_path
        self.coastal_type_raw_data_path = coastal_type_raw_data_path

        if self.download:
            # append year to directory name for organization
            self.directory = os.path.join(self.directory, str(self.year))
            # check for the provided directory and create it if it doesn't exist, or clear it if it does.
            self.check_directory()

    def check_api_key(self):
        """Check for Census API key and warn if not provided, allowing user to choose whether to continue without it."""
        if not self.api_key:
            print("Warning: No Census API key provided. Please set the CENSUS_KEY environment variable and pass it to the DataGetter constructor for faster data downloads.")
            print("Refer to the README for instructions on how to obtain and set your Census API key.")
            print("Do you want to continue without an API key? (y/n)")
            choice = input().lower().strip()
            if choice != 'y':
                print("Exiting.")
                return False
        return True
    
    def check_directory(self):
        """Check if the output directory exists. If it does, delete all files in it. If it doesn't, create it."""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        else:
            # delete all files in the directory if it already exists
            for filename in os.listdir(self.directory):
                file_path = os.path.join(self.directory, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

    def check_file_existence(self, output_file):
        """Check if the output file exists after saving. If it doesn't, raise an error."""
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"File existence check failed: {output_file}")

    def Population_csv(self):
        """
        Download ACS 5-year county-level population data for a given year
        and save it as a CSV.
        """

        url = f"https://api.census.gov/data/{self.year}/acs/acs5"
        params = {
                "get": "NAME,B01003_001E",
                "for": "county:*",
                "key": self.api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])

        df.rename(columns={
                "B01003_001E": "population"
        }, inplace=True)

        return df
    
    def MedianIncome_csv(self):
        """
        Download ACS 5-year county-level median household income for a given year
        and save it as a CSV. The median household income is inflation-adjusted to the given year.

        Output
        ------
        Writes a CSV named:
        county_median_household_income_<year>.csv
        """

        url = f"https://api.census.gov/data/{self.year}/acs/acs5"
        params = {
            "get": "NAME,B19013_001E",
            "for": "county:*",
            "in": "state:*",
            "key": self.api_key
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])

        df.rename(columns={"B19013_001E": "MedianIncome"}, inplace=True)

        return df
    
    def HouseAge_csv(self):
        """
        Downloads ACS 5-year county-level home age data (Table B25034)
        for a given year and saves it as a CSV.
        """

        # ACS 5-year endpoint
        base_url = f"https://api.census.gov/data/{self.year}/acs/acs5"

        # B25034: Year structure built
        variables = [
            "NAME",
            "B25034_001E",  # Total housing units
            "B25034_002E",  # Built 2020 or later (recent years vary by ACS year)
            "B25034_003E",  # Built 2010 to 2019
            "B25034_004E",  # Built 2000 to 2009
            "B25034_005E",  # Built 1990 to 1999
            "B25034_006E",  # Built 1980 to 1989
            "B25034_007E",  # Built 1970 to 1979
            "B25034_008E",  # Built 1960 to 1969
            "B25034_009E",  # Built 1950 to 1959
            "B25034_010E",  # Built 1940 to 1949
            "B25035_001E"   # Median age of housing units
        ]

        params = {
            "get": ",".join(variables),
            "for": "county:*",
            "in": "state:*",
            "key": self.api_key
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()

        df = pd.DataFrame(data[1:], columns=data[0])

        # Convert numeric columns
        for col in variables[1:12]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # rename columns for clarity
        df.rename(columns={
            "B25034_001E": "total_housing_units",
            "B25034_002E": "built_2020_or_later",
            "B25034_003E": "built_2010_to_2019",
            "B25034_004E": "built_2000_to_2009",
            "B25034_005E": "built_1990_to_1999",
            "B25034_006E": "built_1980_to_1989",
            "B25034_007E": "built_1970_to_1979",
            "B25034_008E": "built_1960_to_1969",
            "B25034_009E": "built_1950_to_1959",
            "B25034_010E": "built_1940_to_1949",
            "B25035_001E": "MEDIAN_YEAR_BUILT"
        }, inplace=True)
        
        return df

    def RONI_csv(self):

        # check for raw data file existence and if it does not exist, raise an error
        if not os.path.exists(self.roni_raw_data_path):
            raise FileNotFoundError("The required RONI raw data file is not found")

        # read in the csv file
        df = pd.read_csv(self.roni_raw_data_path)

        # remove rows in yea column with the value "Year"
        df = df[df['Year'] != 'Year']

        # convert values in the dataframe to numeric values
        df = df.apply(pd.to_numeric, errors='coerce')

        # set the year as the index
        df = df.set_index('Year')

        # rename the columns DJF becomes december, january, and february
        df = df.rename(columns={'DJF': 'DECEMBER_JANUARY_FEBRUARY'})
        df = df.rename(columns={'JFM': 'JANUARY_FEBRUARY_MARCH'})
        df = df.rename(columns={'FMA': 'FEBRUARY_MARCH_APRIL'})
        df = df.rename(columns={'MAM': 'MARCH_APRIL_MAY'})
        df = df.rename(columns={'AMJ': 'APRIL_MAY_JUNE'})
        df = df.rename(columns={'MJJ': 'MAY_JUNE_JULY'})
        df = df.rename(columns={'JJA': 'JUNE_JULY_AUGUST'})
        df = df.rename(columns={'JAS': 'JULY_AUGUST_SEPTEMBER'})
        df = df.rename(columns={'ASO': 'AUGUST_SEPTEMBER_OCTOBER'})
        df = df.rename(columns={'SON': 'SEPTEMBER_OCTOBER_NOVEMBER'})
        df = df.rename(columns={'OND': 'OCTOBER_NOVEMBER_DECEMBER'})
        df = df.rename(columns={'NDJ': 'NOVEMBER_DECEMBER_JANUARY'})

        # split each column with the month names into three separate columns with the month name as the column name and the value as the value.
        # if month exists already add the value to the existing column.
        month_columns = df.columns[df.columns.str.contains('_')]

        for month_column in month_columns:
            months = month_column.split('_')
            for month in months:
                col_name = month
                if col_name in df.columns:
                    df[col_name] += df[month_column]
                else:
                    df[col_name] = df[month_column]
        
            # drop the original column
            df = df.drop(columns=[month_column])

        # divide all month columns by 3 to get the average value for each month
        df = df.div(3)

        # get the row corresponding to the given year
        df = df.loc[int(self.year)]

        # reset the index to get the month names back as a column
        df = df.reset_index()

        return df

    def stormDamage_csv(self):
        """
        Download NOAA Storm Events Database damage data for a given year
        and save it as a CSV.
        """

        # Get the URL for the StormEvents details file for the given year
        base_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
        r = requests.get(base_url)
        r.raise_for_status()
        
        pattern = re.compile(
            fr'StormEvents_details-ftp_v1\.0_d{self.year}_c\d+\.csv\.gz'
        )
        match = pattern.search(r.text)
        if match:
            url = str(base_url + match.group(0))
        else:
            raise ValueError(f"No StormEvents details file found for year {self.year}")
        
        # Download and read the CSV file into a df
        df = pd.read_csv(url, compression='gzip', low_memory=False)
        
        return df

    def tempAnomaly_csv(self):
        """
        Downloads monthly county temperature CSVs for the full year from NOAA 
        and merges them into a single file.
        """
        base_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/mapping/110/tavg/"
        all_monthly_data = []

        for month in range(1, 13):
            # Format month to YYYYMM (e.g., 202301)
            date_code = f"{self.year}{str(month).zfill(2)}"
            url = f"{base_url}{date_code}.csv"

            response = requests.get(url)

            # if response is successful, parse the data from memory, otherwise skip
            if response.status_code == 200:
                # Parse CSV content from memory
                csv_content = StringIO(response.text)
                monthly_df = pd.read_csv(csv_content, skiprows=3)  # Skip metadata rows
                monthly_df['MONTH'] = month

                # change column that starts with location id to fips code
                monthly_df['PartialFIPS'] = monthly_df['ID']

                # change value column to temperature
                monthly_df.rename(columns={'Value': 'TEMPERATURE'}, inplace=True)

                all_monthly_data.append(monthly_df[['PartialFIPS','TEMPERATURE', 'Anomaly (1901-2000 base period)', 'MONTH']])
        
        # Merge all monthly DataFrames into a single master DataFrame
        master_df = pd.concat(all_monthly_data, ignore_index=True)

        return master_df

    def coastalType_csv(self):
        """
        2010 shoreline and watershed county data. 
        This is a static dataset that doesn't change by year.
        """

        df = pd.read_csv(self.coastal_type_raw_data_path)
        return df

    def fetch_all(self):
        """
        Run all data getter functions to download and save all datasets.
        """

        # run class attributes ending with _csv to fetch all datasets
        if self.check_api_key():
            population_df = self.Population_csv()
            median_income_df = self.MedianIncome_csv()
            house_age_df = self.HouseAge_csv()
            roni_df = self.RONI_csv()
            storm_data_df = self.stormDamage_csv()
            temp_anomaly_df = self.tempAnomaly_csv()
            coastal_type_df = self.coastalType_csv()

        dataframes = {
                "population": population_df,
                "median_income": median_income_df,
                "house_age": house_age_df,
                "roni": roni_df,
                "storm_data": storm_data_df,
                "temp_anomaly": temp_anomaly_df,
                "coastal_type": coastal_type_df
            }
        
        if self.download:
            for name, df in dataframes.items():
                output_file = f"{self.directory}/{name}.csv"
                df.to_csv(output_file, index=False)
                self.check_file_existence(output_file)
        return dataframes