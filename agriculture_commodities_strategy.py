import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def create_fvg_example():
    """Create 1-minute Live Cattle futures data with FIRST FVG right at session open"""
    
    # Create timestamp starting at 9:30:00 AM EST (session open)
    base_time = datetime(2024, 12, 15, 9, 30, 0)  # Exact 9:30:00 AM start
    times = []
    
    # Generate 2 hours of 1-minute bars (9:30 AM - 11:30 AM)
    for i in range(120):
        times.append(base_time + timedelta(minutes=i))
    
    # Create Live Cattle data where FIRST FVG forms immediately at session open
    np.random.seed(99)  # New seed for immediate FVG
    prices = []
    
    # Start with realistic overnight/pre-market close
    overnight_close = 185.42
    
    for i in range(120):
        if i == 0:  # 9:30:00 AM - First candle after overnight
            current_price = 185.47  # Small gap up from overnight
            open_price = overnight_close + 0.08
        elif i == 1:  # 9:31:00 AM - Second candle (creates the gap)
            current_price = 185.88  # Significant gap up (news/volume)
            open_price = 185.50
        elif i == 2:  # 9:32:00 AM - Third candle (confirms FVG)
            current_price = 185.85  # Slight pullback but gap holds
            open_price = 185.87
        elif i < 30:  # Next 30 minutes - continuation/consolidation
            trend = 0.002
            volatility = 0.04
            current_price += np.random.normal(trend, volatility)
        elif i < 60:  # Second half hour - normal trading
            trend = 0.001  
            volatility = 0.05
            current_price += np.random.normal(trend, volatility)
        else:  # Later session - quieter
            trend = 0.0005
            volatility = 0.03
            current_price += np.random.normal(trend, volatility)
        
        # Keep in realistic range
        current_price = max(min(current_price, 190.00), 182.00)
        
        # Create realistic OHLC
        if i <= 2:  # First 3 candles - controlled for clear FVG
            wick_size = abs(np.random.normal(0, 0.03))
        else:
            open_variance = np.random.normal(0, 0.02)
            open_price = current_price + open_variance
            wick_size = abs(np.random.normal(0, 0.04))
        
        # Create high/low based on candle direction
        if current_price >= open_price:  # Bullish candle
            high = current_price + wick_size
            low = open_price - wick_size * 0.8
        else:  # Bearish candle
            high = open_price + wick_size * 0.8
            low = current_price - wick_size
            
        prices.append({
            'Time': times[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(current_price, 2),
            'Volume': np.random.randint(1200, 3000) if i < 10 else np.random.randint(600, 1800)
        })
    
    return pd.DataFrame(prices)

def plot_fvg_example(df):
    """Plot the 1-minute chart with First FVG highlighted and labeled"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot candlesticks
    for i, row in df.iterrows():
        time = row['Time']
        
        # Determine candle color
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        edge_color = 'darkgreen' if row['Close'] >= row['Open'] else 'darkred'
        
        # Draw candle body
        body_height = abs(row['Close'] - row['Open'])
        body_bottom = min(row['Open'], row['Close'])
        
        ax.bar(time, body_height, bottom=body_bottom, 
               width=timedelta(minutes=0.7), color=color, 
               edgecolor=edge_color, alpha=0.8, linewidth=0.5)
        
        # Draw wicks
        ax.plot([time, time], [row['Low'], row['High']], 
                color='black', linewidth=1, alpha=0.7)
    
    # Identify FVG candles (indices 0, 1, 2 = 9:30, 9:31, 9:32 - THE FIRST FVG!)
    c1_idx, c2_idx, c3_idx = 0, 1, 2
    c1 = df.iloc[c1_idx]
    c2 = df.iloc[c2_idx]  
    c3 = df.iloc[c3_idx]
    
    # Check for Bullish FVG: C1 Low > C3 High
    if c1['Low'] > c3['High']:
        fvg_type = "Bullish"
        fvg_open = c3['High']  # Bottom of FVG
        fvg_close = c1['Low']  # Top of FVG
        fvg_color = 'blue'
    else:
        fvg_type = "Bearish"  
        fvg_open = c1['High']  # Top of FVG
        fvg_close = c3['Low']  # Bottom of FVG
        fvg_color = 'red'
    
    fvg_mid = (fvg_open + fvg_close) / 2
    
    # Draw FVG zone from formation time to end of chart
    zone_start = c3['Time']
    zone_end = df['Time'].iloc[-1]
    
    ax.axhspan(fvg_open, fvg_close, 
               xmin=(zone_start - df['Time'].iloc[0]).total_seconds() / (zone_end - df['Time'].iloc[0]).total_seconds(),
               alpha=0.25, color=fvg_color, zorder=1)
    
    # Draw FVG boundary lines
    ax.axhline(fvg_open, color=fvg_color, linestyle='-', 
               linewidth=2, alpha=0.8)
    ax.axhline(fvg_close, color=fvg_color, linestyle='-', 
               linewidth=2, alpha=0.8)
    ax.axhline(fvg_mid, color='purple', linestyle='--', 
               linewidth=2, alpha=0.7)
    
    # Format x-axis to show every 15 minutes
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add title and labels
    ax.set_title(f'First FVG - Live Cattle Futures', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_ylabel('Price ($/cwt)', fontsize=12)
    ax.set_xlabel('Time (EST)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set y-axis limits to show FVG clearly
    y_min = df['Low'].min() - 0.2
    y_max = df['High'].max() + 0.2
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig

# Create and plot the example
if __name__ == "__main__":
    print("🐄 Creating First FVG Example Chart for Live Cattle Futures...")
    
    # Generate example data
    df = create_fvg_example()
    
    # Print some details
    print(f"Generated {len(df)} 1-minute bars from {df['Time'].iloc[0].strftime('%H:%M')} to {df['Time'].iloc[-1].strftime('%H:%M')} EST")
    
    # Check the FVG manually
    c1, c2, c3 = df.iloc[45], df.iloc[46], df.iloc[47]
    print(f"\nFVG Analysis:")
    print(f"Candle 1 ({c1['Time'].strftime('%H:%M')}): Low=${c1['Low']:.2f}, High=${c1['High']:.2f}")
    print(f"Candle 2 ({c2['Time'].strftime('%H:%M')}): Low=${c2['Low']:.2f}, High=${c2['High']:.2f}")
    print(f"Candle 3 ({c3['Time'].strftime('%H:%M')}): Low=${c3['Low']:.2f}, High=${c3['High']:.2f}")
    print(f"Bullish FVG? {c1['Low'] > c3['High']} (C1 Low: ${c1['Low']:.2f} > C3 High: ${c3['High']:.2f})")
    
    if c1['Low'] > c3['High']:
        print(f"✅ THE FIRST BULLISH FVG CONFIRMED!")
        print(f"   📍 Formed immediately at session open (9:32 AM)")
        print(f"   🎯 FVG Zone: ${c3['High']:.2f} to ${c1['Low']:.2f}")
        print(f"   📏 Gap Size: ${c1['Low'] - c3['High']:.2f}")
        print(f"   🎪 FVG Mid: ${(c3['High'] + c1['Low'])/2:.2f}")
        print(f"   ⏰ Formation Time: {c3['Time'].strftime('%H:%M:%S')} EST")
    
    # Create the visualization
    fig = plot_fvg_example(df)
    plt.show()
