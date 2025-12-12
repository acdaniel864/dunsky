"""
Database utility module for handling database connections and queries.
"""

from sqlalchemy import create_engine, text
import pandas as pd
import datetime
from typing import Optional, List, Dict, Any, Union

from  research_lab.utils.config import DATABASE_URL

def get_engine():
    """
    Creates and returns a SQLAlchemy engine instance.
    This centralizes engine creation and allows for connection pooling.
    
    Returns:
        SQLAlchemy engine instance
    """
    return create_engine(DATABASE_URL)


def get_mp_confidence(engine, lwin_str):
    query = f"""
        SELECT lwin11, market_price_confidence, date
        FROM lx_price
        WHERE lwin11 IN ('{lwin_str}') AND date = (
            SELECT MAX(date) 
            FROM lx_price 
            WHERE lwin11 IN ('{lwin_str}')
        )
    """
    df_confidence = get_data_from_rds(engine, query)
    return df_confidence


def get_data_from_rds(engine, query):
    """
    Execute a SQL query and return the results as a DataFrame.
    
    Args:
        engine: SQLAlchemy engine
        query: SQL query string
        
    Returns:
        DataFrame with query results
    """
    try:
        with engine.connect() as connection:
            # Read the data into a pandas DataFrame
            df = pd.read_sql_query(text(query), connection)
            return df
    except Exception as e:
        print(f"Error pulling data from RDS: {e}")
        return None


def get_df_time(engine, query, price_col_name='price_clean'):
    """
    Executes a query and pivots the results to create a time series DataFrame.
    
    Args:
        query: SQL query string
        engine: Optional SQLAlchemy engine (will create one if not provided)
        
    Returns:
        Pivoted DataFrame with dates as index
    """
    if engine is None:
        engine = get_engine()
        
    df = get_data_from_rds(engine, query)
    if df is None or df.empty:
        return pd.DataFrame()
        
    df_time = df.pivot(index='date', columns='lwin11', values=price_col_name)
    df_time.index = pd.to_datetime(df_time.index, format='%Y-%m-%d')
    return df_time

def save_to_rds(df, engine, region_name, dataset_cat='unknown'):
    """
    Save a DataFrame to an RDS database table.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save to RDS
    table_name : str
        The name of the table to create/replace in RDS
    
    Returns:
    --------
    str
        Path to the CSV file that was also created as backup
    """
    if isinstance(region_name, list):
        table_region = '_'.join(region_name)
    else:
        table_region = str(region_name)
    
    table_name = f"{table_region}_{dataset_cat}"

    # Save DataFrame to RDS
    print(f"Saving data to RDS table: {table_name}")
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Successfully saved {len(df)} rows to RDS table: {table_name}")

    file_name = f"{datetime.datetime.now().strftime('%y-%m-%d')}_{table_name}.csv"
    
    return file_name


def fetch_lx_prices(lwin11_values: Union[str, List[str]], engine=None) -> Optional[pd.DataFrame]:
    """
    Fetch Liv-ex prices from the database for given lwin11 values.
    
    Parameters
    ----------
    lwin11_values : str or List[str]
        Single lwin11 or list of lwin11 identifiers to query
    engine : SQLAlchemy engine, optional
        Database engine. If not provided, a new one will be created.
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: lwin11, date, price, price_per_bottle
        Returns None if query fails or no results found
    """
    if engine is None:
        engine = get_engine()
    
    # Ensure lwin11_values is a list
    if isinstance(lwin11_values, str):
        lwin11_values = [lwin11_values]
    
    if not lwin11_values:
        return None
    
    # Build parameterized query
    placeholders = ', '.join([f"'{v}'" for v in lwin11_values])
    query = f"""
        SELECT lwin11, date, price
        FROM public.lx_price 
        WHERE lwin11 IN ({placeholders})
        ORDER BY lwin11, date
    """
    
    df_lx_prices = get_data_from_rds(engine, query)
    
    if df_lx_prices is None or df_lx_prices.empty:
        return None
    
    # Convert date and calculate price per bottle
    df_lx_prices['date'] = pd.to_datetime(df_lx_prices['date'])
    df_lx_prices['price_per_bottle'] = df_lx_prices['price'] / 12
    
    return df_lx_prices
