import pandas as pd
import numpy as np
import us
import os

class DataProcessor:
    def __init__(self, year, data_dict, data_path = r"./cleanedData/", download=False):
        self.year = year
        self.data_dict = data_dict
        self.data_path = data_path
        self.download = download

        if self.download:
            self.data_path = data_path + f"{year}/"
            # create the directory if it doesn't exist
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)

    def make_fips_code(self, state_code_col, county_code_col):
        # for each row, combine state and county fips and pad with zeros on left to ensure they are 5 digits long. Add a new column called 'FIPS' to the dataframe
        return state_code_col.astype(str).str.zfill(2) + county_code_col.astype(str).str.zfill(3)

    def clean_housing(self):
        df = self.data_dict["house_age"]

        df['FIPS'] = self.make_fips_code(df['state'], df['county'])
        new_df = df.drop(columns=['state', 'county'])

        # find the house age bin with the highest count for each row
        age_columns = [col for col in new_df.columns if col.startswith('built_')]
        new_df['MODAL_YEAR_BUILT_BIN'] = new_df[age_columns].idxmax(axis=1)

        new_df = new_df[['FIPS', 'MODAL_YEAR_BUILT_BIN', 'MEDIAN_YEAR_BUILT']]
        
        return new_df
    
    def clean_storm(self):
        df = self.data_dict["storm_data"]

        # keep columns: first 6 columns, state, state_fips, month_name, event_type,cz_fips cz_name, damage_property, begin_lat, begin_lon, end_lat, end_lon. All names are in upper case
        new_df = df.iloc[:, :6]
        join_df = df[['STATE', 'STATE_FIPS', 'MONTH_NAME', 'EVENT_TYPE', 'CZ_FIPS', 'CZ_NAME', 'DAMAGE_PROPERTY', 'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON', 'BEGIN_DAY']]
        new_df = pd.concat([new_df, join_df], axis=1)



        # calculate the approximate area of the storm by using the haversine formula to calculate the distance between the begin and end coordinates. Add a new column called 'STORM_AREA' to the dataframe
        # drop na values in cordinates columns
        new_df = new_df.dropna(subset=['BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON'])
        def rectagular_area(begin_lat, begin_lon, end_lat, end_lon):
            # convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [begin_lat, begin_lon, end_lat, end_lon])

            # use A = R² (sin lat1 − sin lat2) (lon1 − lon2).
            # from https://www.johndcook.com/blog/2023/02/21/sphere-grid-area/#:~:text=Area%20of%20latitude/longitude%20grid&text=A%20=%20π%20R²%20(sin%20φ,1%20−%20θ2)/180.
            r = 3956  # Radius of earth in miles
            area = r**2 * (np.sin(lat1) - np.sin(lat2)) * (lon1 - lon2)
            return abs(area)
        new_df['STORM_AREA_SQMILES'] = new_df.apply(lambda row: rectagular_area(row['BEGIN_LAT'], row['BEGIN_LON'], row['END_LAT'], row['END_LON']), axis=1)

        # drop the begin and end lat and lon columns
        new_df = new_df.drop(columns=['BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON'])


        # calculate the total damage by converting the damage property column to a numeric value. 
        # The damage property column is in the format of a string with a number followed by a letter (K, M, B) which represents the magnitude of the damage. 
        # keep missing values
        def convert_damage(damage):
            if pd.isna(damage):
                return np.nan
            elif damage.endswith('K'):
                return float(damage[:-1]) * 1e3
            elif damage.endswith('M'):
                return float(damage[:-1]) * 1e6
            elif damage.endswith('B'):
                return float(damage[:-1]) * 1e9
        new_df['DAMAGE_PROPERTY'] = new_df['DAMAGE_PROPERTY'].apply(convert_damage)
        
        # calculate the duration of the storm by using begin time and end time columns which are in military time (hhmm)
        def calculate_duration(row):
            begin_time = row['BEGIN_TIME']
            end_time = row['END_TIME']

            # pad the time strings with zeros if they are less than 4 characters long
            begin_time = str(begin_time).zfill(4)
            end_time = str(end_time).zfill(4)

            begin_hours = int(begin_time[:2]) 
            end_hours = int(end_time[:2]) 
            begin_minutes = int(begin_time[2:])
            end_minutes = int(end_time[2:])

            duration = (end_hours * 60 + end_minutes) - (begin_hours * 60 + begin_minutes)
            if duration < 0:
                duration += 24 * 60  # Adjust for storms that last past midnight
            return duration  # Return duration in minutes
        new_df['DURATION_MINUTES'] = new_df.apply(calculate_duration, axis=1)



        # keep event types: ones with flood in the name, Hail, heavy rain, high wind, lightning, strong wind, thunderstorm wind, and tornado
        new_df = new_df[new_df['EVENT_TYPE'].str.contains('FLOOD|HAIL|HEAVY RAIN|HIGH WIND|LIGHTNING|STRONG WIND|THUNDERSTORM WIND|TORNADO', case=False, na=False)]
        # drop marine event types
        new_df = new_df[~new_df['EVENT_TYPE'].str.contains('Marine', case=False, na=False)]



        # get month from yearmonth column and add it as a new column called 'MONTH'
        new_df['MONTH'] = new_df['BEGIN_YEARMONTH'].astype(str).str[4:6].astype(int)
        # drop first 6 columns
        new_df = new_df.drop(columns=new_df.columns[:6])




        # combine state cz fips and pad with zeros on left to ensure they are 5 digits long. Add a new column called 'FIPS' to the dataframe
        new_df['FIPS'] = self.make_fips_code(df['STATE_FIPS'], df['CZ_FIPS'])
        # drop STATE_FIPS and CZ_FIPS columns
        new_df = new_df.drop(columns=['STATE_FIPS', 'CZ_FIPS'])



        # keep only states in the continental US
        non_continental_states = ['ALASKA', 'HAWAII', 'PUERTO RICO', 'GUAM', 'VIRGIN ISLANDS', 'AMERICAN SAMOA', 'NORTHERN MARIANA ISLANDS'] # keeping the district of columbia
        new_df = new_df[~new_df['STATE'].isin(non_continental_states)]

        # capitalize MONTH_NAME
        new_df['MONTH_NAME'] = new_df['MONTH_NAME'].str.upper()

        return new_df
    
    def clean_population(self):
        df = self.data_dict["population"]

        # combine state and county fips and pad with zeros on left to ensure they are 5 digits long. Add a new column called 'FIPS' to the dataframe
        df['FIPS'] = self.make_fips_code(df['state'], df['county'])
        # drop STATE_FIPS and COUNTY_FIPS columns
        new_df = df.drop(columns=['state', 'county'])

        # drop all columns except FIPS and population
        new_df = new_df[['FIPS', 'population']]

        # capitalize the population column name
        new_df = new_df.rename(columns={'population': 'POPULATION'})

        return new_df
    
    def clean_income(self):
        df = self.data_dict["median_income"]

        # combine state and county fips and pad with zeros on left to ensure they are 5 digits long. Add a new column called 'FIPS' to the dataframe
        df['FIPS'] = self.make_fips_code(df['state'], df['county'])

        # drop STATE_FIPS and COUNTY_FIPS columns
        new_df = df.drop(columns=['state', 'county'])

        # convert median income column to numeric and coerce errors to NaN
        new_df['MedianIncome'] = pd.to_numeric(new_df['MedianIncome'], errors='coerce')

        # drop negative income values
        new_df = new_df[new_df['MedianIncome'] >= 0]

        # drop all columns except FIPS and median income
        new_df = new_df[['FIPS', 'MedianIncome']]

        # capitalize the median income column name
        new_df = new_df.rename(columns={'MedianIncome': 'MEDIAN_INCOME'})
        
        return new_df
    
    def clean_roni(self):
        df = self.data_dict["roni"]

        # set the first column to Month
        df = df.rename(columns={df.columns[0]: 'MONTH_NAME'})

        df = df.rename(columns = {df.columns[1]: 'RONI_AVG'})

        return df
    
    def clean_anomaly(self):
        # read in the csv file
        df = self.data_dict["temp_anomaly"]

        # split partial fips data into two columns on "-"
        df[['STATE_ABBR', 'COUNTY_FIPS']] = df[df.columns[0]].str.split('-', expand=True)

        # drop partial fips and state abbreviation column
        df = df.drop(columns=[ 'PartialFIPS'])

        # convert state abreviation to state number using a mapping dictionary
        def abbreviation_to_fips(abbreviation):
            """Converts a state abbreviation to its FIPS number using the 'us' package."""
            state = us.states.lookup(abbreviation)
            if state:
                # state.fips returns the FIPS code as a string, convert to int if needed
                return int(state.fips)
            else:
                return "Invalid abbreviation"
            
        df['STATE_FIPS'] = df['STATE_ABBR'].apply(abbreviation_to_fips)

        # drop state abbreviation column
        df = df.drop(columns=['STATE_ABBR'])

        # combine state and county fips and pad with zeros on left to ensure they are 5 digits long. Add a new column called 'FIPS' to the dataframe
        df['FIPS'] = self.make_fips_code(df['STATE_FIPS'], df['COUNTY_FIPS'])
        df = df.drop(columns=['STATE_FIPS', 'COUNTY_FIPS'])

        # change the column starting with Anomaly to just ANOMALY
        df = df.rename(columns= {[col for col in df.columns if col.startswith('Anomaly')][0]: 'ANOMALY_F'})

        # add fahrenheit to end of temperature column name
        df = df.rename(columns={"TEMPERATURE": 'TEMPERATURE_F'})

        return df
    
    def clean_coastalTypes(self):
        df = self.data_dict["coastal_type"]

        # pad fips with zeros on left to ensure they are 5 digits long.
        df['FIPS'] = df['FIPS'].apply(lambda x: str(x).zfill(5))

        # fill missing values in COASTAL_TYPE_SHORELINE with inland
        df["COASTAL_TYPE_SHORELINE"] = df["COASTAL_TYPE_SHORELINE"].fillna("inland")

        # drop columns starting with NAME
        df = df.drop(columns=[col for col in df.columns if col.startswith("NAME")], errors='ignore')

        return df
    
    def process_all(self):
        """run all data cleaning methods."""
        population_df = self.clean_population()
        median_income_df = self.clean_income()
        house_age_df = self.clean_housing()
        roni_df = self.clean_roni()
        storm_data_df = self.clean_storm()
        temp_anomaly_df = self.clean_anomaly()
        coastal_type_df = self.clean_coastalTypes()

        cleaned_data = {
                "population": population_df,
                "median_income": median_income_df,
                "housing": house_age_df,
                "roni": roni_df,
                "storm_data": storm_data_df,
                "temp_anomaly": temp_anomaly_df,
                "coastal_type": coastal_type_df
        }

        if self.download:
            for key, df in cleaned_data.items():
                df.to_csv(self.data_path + f"cleaned_{key}.csv", index=False)

        return cleaned_data