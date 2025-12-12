import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates   


diverse_colors = ["#9437ff", "#83D483", "#FFD166", "#F78C6B", 
                        "#4D87D0", "#EF476F", "#06D6A0", "#C23FB7", 
                        "#4A4A68"]
watermark_path = 'images/watermark600x600.png'

def plot_regional_indexes_plotly(df, title, watermark_path):
    colors = [
        '#9437ff',  # purple
        '#83D483',  # mantis
        '#FFD166',  # sunglow
        '#F78C6B',  # coral
        '#4D87D0',  # blue
        '#EF476F',  # red
        '#06D6A0',  # emerald
        '#C23FB7',  # pink/purple
        '#4A4A68'   # slate
    ]
    
    fig = go.Figure()
    for i, region in enumerate(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df[region], 
                mode="lines",
                name=region,
                line=dict(color=colors[i % len(colors)])  # Cycle through colors if more regions than colors
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Datetime",
        yaxis_title="Index",
        legend_title="Region",
        hovermode="x unified",
        height=750
    )
    # Include watermark
    watermark = Image.open(watermark_path)
    fig.add_layout_image(
        dict(
            source=watermark,
            xref="paper", yref="paper",
            x=0.52, y=0.5,  # Positioning the logo at the center
            xanchor='center', yanchor='middle',
            sizex=0.6, sizey=0.6,  # Adjust the size of the logo
            opacity=0.65,
            layer="above"
        ))

    fig.show()

def filter_most_expensive_wines(df, price_max_threshold=100000):
    # Define the target number of wines per region
    region_targets = {
        'Bordeaux': 2323,
        'Burgundy': 1345,
        'Champagne': 672,
        'Tuscany': 326,
        'Piedmont': 652,
        'Rhone': 183,
        'California': 61
    }
    other_target = 224.5
    
    # Drop rows with missing current_price or region
    df = df.dropna(subset=['current_price', 'region'])
    
    # Apply maximum price threshold
    df = df[df['current_price'] <= price_max_threshold]
    
    # Create a boolean column to identify rows that belong to OTHER regions
    df['region_group'] = df['region'].apply(lambda x: x if x in region_targets else 'OTHER')
    
    # Initialize an empty DataFrame to store the results
    filtered_df = pd.DataFrame()
    
    # Filter the most expensive wines for each specified region
    for region, target in region_targets.items():
        region_df = df[df['region_group'] == region]
        most_expensive_region = region_df.nlargest(int(target), 'current_price')
        filtered_df = pd.concat([filtered_df, most_expensive_region])
    
    # Handle the OTHER regions
    other_df = df[df['region_group'] == 'OTHER']
    most_expensive_other = other_df.nlargest(int(other_target), 'current_price')
    filtered_df = pd.concat([filtered_df, most_expensive_other])
    
    # Drop the auxiliary 'region_group' column
    filtered_df = filtered_df.drop(columns=['region_group'])

    print(filtered_df['region'].value_counts())
    
    return filtered_df

def make_true_price_weight_index(df, smoothing_months=None):
    """
    Create a price-weighted index for wine prices.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series price data with lwin11 as columns
    smoothing_months (int, optional): Number of months for rolling average smoothing. 
                                    If None, no smoothing is applied.
    
    Returns:
    pd.DataFrame: DataFrame with the calculated returns, normalized index, and additional metrics
    """
    # Forward fill NaN values
    df = df.ffill()
    
    # Create a new DataFrame for calculations
    new_df = pd.DataFrame(index=df.index)
    
    # Calculate percentage change for each wine
    pct_changes = df.pct_change()
    
    # Calculate the price-weighted average percentage change
    def price_weighted_change(row, prev_row):
        weights = prev_row.dropna()
        changes = row[weights.index]
        return (weights * changes).sum() / weights.sum()
    
    new_df['weighted_pct_change'] = [price_weighted_change(pct_changes.iloc[i], df.iloc[i-1]) 
                                     if i > 0 else np.nan 
                                     for i in range(len(df))]
    
    # Create normalized index starting at 100 (unsmoothed)
    new_df['index_value'] = (1 + new_df['weighted_pct_change']).cumprod() * 100
    new_df['index_value'] = new_df['index_value'].fillna(100)
    
    # Additional informational columns
    new_df['qty_active_wines'] = df.notna().sum(axis=1)
    new_df['total_price'] = df.sum(axis=1)
    new_df['mean_price'] = new_df['total_price'] / new_df['qty_active_wines']
    
    # Apply smoothing if specified and add smoothed version
    if smoothing_months is not None:
        new_df['index_value_smoothed'] = new_df['index_value'].rolling(window=smoothing_months, min_periods=1).mean()
    
    return new_df

def make_portfolio_weighted_index(df, portfolio_weights, smoothing_months=None):
    """
    Create a portfolio-weighted index where each wine is weighted by the number of bottles held.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series price data with lwin11 as columns
    portfolio_weights (dict): Dictionary mapping lwin11 to number of bottles held
                             e.g., {'lwin11_1': 12, 'lwin11_2': 3, 'lwin11_3': 8}
    smoothing_months (int, optional): Number of months for rolling average smoothing.
                                    If None, no smoothing is applied.
    
    Returns:
    pd.DataFrame: DataFrame with the calculated returns, normalized index, and additional metrics
    """
    # Forward fill NaN values
    df = df.ffill()
    
    # Filter dataframe to only include wines in the portfolio
    portfolio_wines = [wine for wine in portfolio_weights.keys() if wine in df.columns]
    df_portfolio = df[portfolio_wines].copy()
    
    # Create a new DataFrame for calculations
    new_df = pd.DataFrame(index=df_portfolio.index)
    
    # Calculate percentage change for each wine
    pct_changes = df_portfolio.pct_change()
    
    # Calculate the portfolio-weighted average percentage change
    def portfolio_weighted_change(row, prev_row):
        # Get wines that have data in both periods
        active_wines = prev_row.dropna().index
        changes = row[active_wines]
        
        # Calculate portfolio value weights (price * quantity)
        portfolio_values = {}
        total_portfolio_value = 0
        
        for wine in active_wines:
            if wine in portfolio_weights:
                wine_value = prev_row[wine] * portfolio_weights[wine]
                portfolio_values[wine] = wine_value
                total_portfolio_value += wine_value
        
        if total_portfolio_value == 0:
            return np.nan
        
        # Calculate weighted return
        weighted_return = 0
        for wine in active_wines:
            if wine in portfolio_values and not pd.isna(changes[wine]):
                weight = portfolio_values[wine] / total_portfolio_value
                weighted_return += weight * changes[wine]
        
        return weighted_return
    
    new_df['weighted_pct_change'] = [portfolio_weighted_change(pct_changes.iloc[i], df_portfolio.iloc[i-1]) 
                                     if i > 0 else np.nan 
                                     for i in range(len(df_portfolio))]
    
    # Create normalized index starting at 100
    new_df['index_value'] = (1 + new_df['weighted_pct_change']).cumprod() * 100
    new_df['index_value'] = new_df['index_value'].fillna(100)
    
    # Additional informational columns
    new_df['qty_active_wines'] = df_portfolio.notna().sum(axis=1)
    
    # Calculate portfolio metrics
    def calculate_portfolio_value(row):
        total_value = 0
        for wine in portfolio_wines:
            if wine in row.index and not pd.isna(row[wine]) and wine in portfolio_weights:
                total_value += row[wine] * portfolio_weights[wine]
        return total_value
    
    new_df['total_portfolio_value'] = df_portfolio.apply(calculate_portfolio_value, axis=1)
    new_df['total_bottles'] = sum(portfolio_weights[wine] for wine in portfolio_wines if wine in portfolio_weights)
    new_df['avg_price_per_bottle'] = new_df['total_portfolio_value'] / new_df['total_bottles']
    
    # Apply smoothing if specified and add smoothed version
    if smoothing_months is not None:
        new_df['index_value_smoothed'] = new_df['index_value'].rolling(window=smoothing_months, min_periods=1).mean()
    
    return new_df


def plot_index_enhanced(returns, watermark_path, main_purple, qty_active_wines=None, title='', note='Underlying data based on Liv-ex Market prices', label='Index', height=600):
    """Enhanced plotly version with custom styling"""
    fig = go.Figure()

    # Main trace with custom styling
    fig.add_trace(go.Scatter(
        x=returns.index, 
        y=returns, 
        mode='lines', 
        name=label,
        line=dict(color=main_purple, width=4)
    ))
    
    # Add secondary axis if qty_active_wines is provided
    if qty_active_wines is not None:
        fig.add_trace(go.Scatter(
            x=qty_active_wines.index, 
            y=qty_active_wines, 
            mode='lines', 
            name='Active Wines Quantity',
            line=dict(color='#FF6347', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            yaxis2=dict(
                title='Qty Active Wines',
                overlaying='y',
                side='right',
                showgrid=False,
                tickformat=",",
                title_font=dict(family='Avenir, Arial, sans-serif', size=12),
                title_standoff=20
            )
        )
    
    # Enhanced layout styling
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=24, family='Avenir, Arial, sans-serif')
        ),
        xaxis=dict(
            tickformat='%Y',
            dtick='M12',
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor=main_purple,
            tickangle=315
        ),
        yaxis=dict(
            title='Index Value',
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E5E5',
            showline=True,
            linewidth=2,
            linecolor=main_purple,
            title_standoff=20
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Avenir, Arial, sans-serif', size=15),
        legend_title_text='Region',
        width=1000,
        height=height,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Add note with custom styling
    if note:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.00,
            text=note,
            showarrow=False,
            font=dict(size=11, color=main_purple, family='Avenir, Arial, sans-serif'),
            align="left"
        )
    if watermark_path is not None: 
        # Include watermark
        watermark = Image.open(watermark_path)
        fig.add_layout_image(
            dict(
                source=watermark,
                xref="paper", yref="paper",
                x=0.52, y=0.5,  # Positioning the logo at the center
                xanchor='center', yanchor='middle',
                sizex=0.8, sizey=0.8,  # Adjust the size of the logo
                opacity=0.65,
                layer="above"
            ))
        
    fig.show()


def make_equal_weight_index_smoothed_2(df, window=3):
    """
    Create an equal weight index from unbalanced time series price data with smoothing,
    accounting for new wines joining the index.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series price data.
    window (int): The rolling window size for smoothing the effect of new wines.

    Returns:
    pd.DataFrame: DataFrame with the calculated returns, normalized index, and additional metrics.
    """
    # Step 1: Forward fill NaN values
    df_filled = df.ffill()

    # Step 2: Calculate percentage change for each wine
    pct_changes = df_filled.pct_change()

    # Step 3: Calculate the equal-weighted return for each date
    def equal_weighted_change(row, prev_row):
        active_prev = prev_row.dropna()
        active_current = row[active_prev.index]
        return active_current.mean()

    equal_weighted_return = pd.Series([equal_weighted_change(pct_changes.iloc[i], df_filled.iloc[i-1]) 
                                       if i > 0 else np.nan 
                                       for i in range(len(df_filled))],
                                      index=df_filled.index)

    # Step 4: Smooth the equal-weighted returns using a rolling window
    smoothed_return = equal_weighted_return.rolling(window=window, min_periods=1).mean()

    # Step 5: Create a result DataFrame
    result_df = pd.DataFrame({
        'raw_return': equal_weighted_return,
        'smoothed_return': smoothed_return
    })

    # Step 6: Make normalized index starting at 100
    result_df['index_value'] = (1 + result_df['smoothed_return']).cumprod() * 100
    result_df['index_value'] = result_df['index_value'].fillna(100)

    # Step 7: Calculate additional metrics
    result_df['qty_active_wines'] = df.notna().sum(axis=1)
    result_df['total_price'] = df_filled.sum(axis=1)
    result_df['mean_price'] = result_df['total_price'] / result_df['qty_active_wines']

    return result_df




def plot_indices_plotly(returns_dict, watermark_path=watermark_path, colours=diverse_colors, title='', note=''):
    # Color palette in priority order

    # Create the figure
    fig = go.Figure()

    # Add traces for each series with line width 3 and custom colours
    for idx, (label, returns) in enumerate(returns_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns.values,
                mode='lines',
                name=label,
                line=dict(color=colours[idx % len(colours)], width=3)
            )
        )

    # Update layout for better aesthetics
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'Avenir', 'size': 20}
        },
        xaxis_title='Date',
        yaxis_title='Index Value',
        xaxis=dict(
            showgrid=False,
            tickformat='%Y',
            title_font={'family': 'Avenir'}
        ),
        yaxis=dict(
            showgrid=True,
            title_font={'family': 'Avenir'}
        ),
        legend=dict(
            yanchor="bottom",
            y=0.5,
            xanchor="center",
            x=1.08,
            font={'family': 'Avenir'}
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        template='plotly_white',
        font=dict(family='Avenir'),
        annotations=[
            dict(
                text=note,
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.01,
                showarrow=False,
                font=dict(size=12, color="gray"),
                align="left"
            )
        ]
    )


    # Include watermark
    watermark = Image.open(watermark_path)
    fig.add_layout_image(
        dict(
            source=watermark,
            xref="paper", yref="paper",
            x=0.52, y=0.5,  # Positioning the logo at the center
            xanchor='center', yanchor='middle',
            sizex=0.8, sizey=0.8,  # Adjust the size of the logo
            opacity=0.65,
            layer="above"
        ))

    # Show the figure
    fig.show()



def plot_indices(returns_dict, title='', note='', figsize=(8, 5)):    
    plt.figure(figsize=figsize)
    
    for label, returns in returns_dict.items():
        plt.plot(returns.index, returns, label=label)
    
    plt.title(f'{title}')
    plt.xlabel('Date')
    plt.ylabel('Index Value')

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()  # Rotation

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.text(0.01, 0.01, note , transform=plt.gca().transAxes,
             fontsize=9, va='bottom', ha='left', color='#9437FF')

    plt.show()
    

def plot_indices_enhanced(returns_dict, title='', note='', figsize=(10, 6)):
    """Enhanced matplotlib version with custom styling"""
    plt.figure(figsize=figsize)
    
    # Custom color palette matching the first example
    diverse_colors = ['#4B0082', '#FF6347', '#4682B4', '#32CD32', '#FFD700', '#FF69B4']
    
    # Set the background color
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    
    # Plot each line with enhanced styling
    for i, (label, returns) in enumerate(returns_dict.items()):
        plt.plot(returns.index, returns, 
                label=label, 
                linewidth=3,
                color=diverse_colors[i % len(diverse_colors)])
    
    # Title and labels with custom font
    plt.title(title, fontsize=24, pad=20, fontfamily='sans-serif')
    plt.xlabel('Date', fontsize=12, fontfamily='sans-serif')
    plt.ylabel('Index Value', fontsize=12, fontfamily='sans-serif')

    # X-axis formatting
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    # Grid styling
    plt.grid(True, which='major', linestyle='-', linewidth=1, color='#E5E5E5')
    
    # Border styling
    for spine in plt.gca().spines.values():
        spine.set_color('#4B0082')
        spine.set_linewidth(2)

    # Legend styling
    plt.legend(title='Region', frameon=True, fontsize=10)

    # Add note with custom styling
    if note:
        plt.text(0.01, 0.01, note, transform=plt.gca().transAxes,
                fontsize=9, va='bottom', ha='left', color='#4B0082',
                fontfamily='sans-serif')

    plt.tight_layout()
    plt.show()


def plot_index_enhanced(returns, qty_active_wines=None, title='', note='', label='Index', height=600, watermark_path=None):
    """Enhanced plotly version with custom styling"""
    fig = go.Figure()
    diverse_colors = ["#9437ff", "#83D483", "#FFD166", "#F78C6B", 
                          "#4D87D0", "#EF476F", "#06D6A0", "#C23FB7", 
                          "#4A4A68"]
    
    # Main trace with custom styling
    fig.add_trace(go.Scatter(
        x=returns.index, 
        y=returns, 
        mode='lines', 
        name=label,
        line=dict(color='#9437ff', width=3)
    ))
    
    # Add secondary axis if qty_active_wines is provided
    if qty_active_wines is not None:
        fig.add_trace(go.Scatter(
            x=qty_active_wines.index, 
            y=qty_active_wines, 
            mode='lines', 
            name='Active Wines Quantity',
            line=dict(color='#FF6347', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            yaxis2=dict(
                title='Qty Active Wines',
                overlaying='y',
                side='right',
                showgrid=False,
                tickformat=",",
                title_font=dict(family='Avenir, Arial, sans-serif', size=12),
                title_standoff=20
            )
        )
    
    # Enhanced layout styling
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=24, family='Avenir, Arial, sans-serif')
        ),
        xaxis=dict(
            tickformat='%Y',
            dtick='M12',
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='#9437ff',
            tickangle=315
        ),
        yaxis=dict(
            title='Index Value',
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E5E5',
            showline=True,
            linewidth=2,
            linecolor='#9437ff',
            title_standoff=20
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Avenir, Arial, sans-serif', size=12),
        legend_title_text='Region',
        width=1000,
        height=height,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Add note with custom styling
    if note:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.00,
            text=note,
            showarrow=False,
            font=dict(size=11, color='#9437ff', family='Avenir, Arial, sans-serif'),
            align="left"
        )

    # Include watermark
    if watermark_path is not None:
        watermark = Image.open(watermark_path)
        fig.add_layout_image(
            dict(
            source=watermark,
            xref="paper", yref="paper",
            x=0.52, y=0.5,  # Positioning the logo at the center
            xanchor='center', yanchor='middle',
            sizex=0.6, sizey=0.6,  # Adjust the size of the logo
            opacity=0.65,
            layer="above"
        ))

    #
    
    fig.show()
