import requests
import pandas as pd
import json
from datetime import datetime

possible_index_names = ["Liv-ex Fine Wine 50", "Liv-ex Fine Wine 1000", "Burgundy 150", "Champagne 50",  
                        "Liv-ex Fine Wine 100", "Italy 100" , "California 50",
                        "Liv-ex Bordeaux 500", "Bordeaux Legends 50", "Rest of the World 60",
                         "Rhone 100", "Port 50" ]

def convert_wine_data_to_df(json_data):
    """
    Convert Liv-ex wine index JSON data to a pandas DataFrame with properly formatted dates.
    
    Parameters:
    json_data (dict): JSON response from Liv-ex API containing wine index data
    
    Returns:
    pandas.DataFrame: DataFrame with dates and values
    """
    try:
        # Extract the values from the nested JSON structure
        values = json_data['dataSeries']['groups'][0]['list'][0]['series']['values']
        
        # Create a list of dictionaries with converted dates
        data_list = []
        for entry in values:
            data_list.append({
                'date': datetime.fromtimestamp(entry['date']/1000).strftime('%Y-%m-%d'),
                'value': entry['value']
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Add index name as column
        index_name = json_data['dataSeries']['groups'][0]['list'][0]['indexName']
        df['index_name'] = index_name
        
        # Reorder columns
        df = df[['date', 'index_name', 'value']]
        
        return df
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def fetch_liv_ex_data(index_name, timeframe =  "maximum"):
    # API endpoint
    url = 'https://api.liv-ex.com/data/v1/dataSeries'
    
    index = index_name

    # Headers based on the request information
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'client_key': '961f06bb-0489-4c3d-a668-c39cf2c36f1d',
        'client_secret': 'c0nuzzNQ',
        'Origin': 'https://app.liv-ex.com',
        'Referer': 'https://app.liv-ex.com/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
    }
    
    # The exact payload structure from your request
    payload = {
        "dataSeries": {
            "timeframe": timeframe,
            "lwin": [],
            "internalIndex": [index],
            "thirdPartyIndex": [],
            "currency": "GBP",
            "rebase": False
        }
    }
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse and return the JSON response
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def main(index_name, granularity="maximum"):
    if index_name in possible_index_names:
        # Fetch the data
        data = fetch_liv_ex_data(index_name, granularity)
        
        # Check if we got data back
        if data:
            df = convert_wine_data_to_df(data)
            return df
    else: 
        print(f"Correct index names are: {possible_index_names}")
        
if __name__ == "__main__":
    main()
