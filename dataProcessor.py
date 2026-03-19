class DataProcessor:
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def make_fips_code(self, state_code, county_code):
        return str(state_code).zfill(2) + str(county_code).zfill(3)