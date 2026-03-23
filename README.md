# Storm Damage Analysis - Storm Level

A Python-based data pipeline for analyzing storm damage at the storm level, combining data from multiple sources including NOAA storm events, census ACS 5-year API, NCEI climate data, and NOAA's GeoPlatform.

## Project Overview

This project processes and merges multiple datasets to create a comprehensive analysis dataset linking storm events with socioeconomic and environmental factors. The pipeline:

1. **Fetches raw data** from various sources (NOAA, NWS, NCEI, and US Census Bureau)
2. **Cleans and processes** each dataset independently
3. **Merges** all datasets on geographic identifiers (FIPS codes) and the appropriate temporal dimension
4. **Produces** a unified dataset ready for analysis

## Key Features

- Data fetching from Census API
- Calculation of Variables using NumPy
- Data cleaning and standardization across multiple sources using Pandas
- Intelligent merging with FIPS codes, month name, and month numbers matching
- Organized output structure with year-based directories

## Data Sources and Variable Descriptions
[Variable Descriptions PDF](VarDescriptions.pdf)

## Required Libraries

```
pandas
numpy
requests
us
re
os
StringIO
```

Install dependencies with:
```bash
pip install pandas numpy requests us
```

## Getting a Census API Key and Setting an Environment Variable

1. **Obtain a Census API Key**: Visit [https://api.census.gov/data/key_signup.html](https://api.census.gov/data/key_signup.html) to request your free API key.

2. **Set the environment variable**:
   - **Windows (PowerShell)**:
     ```powershell
     $env:CENSUS_KEY = "your_api_key_here"
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     set CENSUS_KEY=your_api_key_here
     ```
   - **macOS/Linux**:
     ```bash
     export CENSUS_KEY="your_api_key_here"
     ```
  
   - **bash**
      ```
      conda env config vars set API_KEY_NAME=your_api_key_value
      ```

3. **Verify the configuration**: In Python, verify with:
   ```python
   import os
   CENSUS_KEY = os.getenv("CENSUS_KEY")
   print(CENSUS_KEY)
   ```