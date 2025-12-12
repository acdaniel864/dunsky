import os
import requests
import pandas as pd

def download_hpi_data():
    """Download UK House Price Index data from the Land Registry"""
    from datetime import datetime, timedelta
    
    target_dir = "data/non_wine_timeseries"
    os.makedirs(target_dir, exist_ok=True)
    
    # Start from current month and work backwards to find the most recent available data
    current_date = datetime.now()
    
    for months_back in range(6):  # Try up to 6 months back
        # Calculate the target month
        target_date = current_date - timedelta(days=30 * months_back)
        year_month = target_date.strftime("%Y-%m")
        
        # Construct URL and filename
        url = f"https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-{year_month}.csv?utm_medium=GOV.UK&utm_source=datadownload&utm_campaign=full_fil&utm_term=9.30_16_10_24"
        target_file = os.path.join(target_dir, f"UK-HPI-full-file-most-recent.csv")
        
        # Check if we already have this file
        if os.path.exists(target_file):
            print(f"UK HPI data already exists for {year_month}.")
            return target_file
        
        print(f"Trying to download UK House Price Index data for {year_month}...")
        
        try:
            # First, check if the URL exists with a HEAD request
            head_response = requests.head(url)
            if head_response.status_code == 200:
                # URL exists, proceed with download
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(target_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Downloaded UK HPI data to: {target_file}")
                return target_file
            else:
                print(f"Data for {year_month} not available (status: {head_response.status_code})")
                
        except Exception as e:
            print(f"Error checking/downloading UK HPI data for {year_month}: {e}")
            continue
    
    print("Could not find any available UK HPI data in the last 6 months")
    return None

if __name__ == "__main__":
    download_hpi_data() 