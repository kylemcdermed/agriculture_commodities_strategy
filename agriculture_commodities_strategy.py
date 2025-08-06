import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

class CompleteFVGTradingSystem:
    def __init__(self, sma_period=7):
        """Complete FVG Trading System with Entry/Exit Logic"""
        self.sma_period = sma_period
        self.data_1m = None
        self.daily_data = None
        self.daily_bias = None
        self.first_fvg = None
        self.entry_signal = None
        self.stop_loss = None
        
    def generate_daily_data(self):
        """Generate daily data for 7-SMA trend bias"""
        # Generate 30 days of daily data
        dates = pd.date_range(start=datetime.now() - timedelta(days=40), end=datetime.now(), freq='D')
        
        np.random.seed(88)
        base_price = 185.00
        daily_prices = []
        
        for i, date in enumerate(dates):
            if i == 0:
                price = base_price
            else:
                # Create trending daily movement
                trend = 0.02 if i < 25 else -0.01  # Uptrend then slight down
                price = daily_prices[-1] + np.random.normal(trend, 0.3)
                price = max(price, 180.00)
            
            # Create OHLC
            high = price + abs(np.random.normal(0, 0.4))
            low = price - abs(np.random.normal(0, 0.4))
            open_price = price + np.random.normal(0, 0.2)
            
            daily_prices.append(price)
            
        self.daily_data = pd.DataFrame({
            'Date': dates,
            'Open': [p + np.random.normal(0, 0.1) for p in daily_prices],
            'High': [p + abs(np.random.normal(0, 0.3)) for p in daily_prices],
            'Low': [p - abs(np.random.normal(0, 0.3)) for p in daily_prices],
            'Close': daily_prices
        })
        
        # Calculate 7-SMA and current bias
        self.daily_data['SMA_7'] = self.daily_data['Close'].rolling(window=self.sma_period).mean()
        latest = self.daily_data.iloc[-1]
        self.daily_bias = 'Bullish' if latest['Close'] > latest['SMA_7'] else 'Bearish'
        
        print(f"Daily Bias: {self.daily_bias} (Close: ${latest['Close']:.2f} vs 7-SMA: ${latest['SMA_7']:.2f})")

    def create_fvg_with_entry_example(self):
        """Create 1-minute data with FVG and entry scenario"""
        base_time = datetime(2024, 12, 15, 9, 30, 0)
        times = [base_time + timedelta(minutes=i) for i in range(120)]
        
        np.random.seed(200)
        prices = []
        overnight_close = 185.25
        
        # Create bullish FVG scenario (assuming bullish daily bias)
        for i in range(120):
            if i == 0:  # 9:30:00 - First candle
                current_price = 185.32
                open_price = overnight_close + 0.05
            elif i == 1:  # 9:31:00 - Gap up candle
                current_price = 185.78  # Big gap up
                open_price = 185.35
            elif i == 2:  # 9:32:00 - Third candle (slight pullback)
                current_price = 185.74
                open_price = 185.77
            elif i == 3:  # 9:33:00 - Consolidation
                current_price = 185.68
                open_price = 185.72
            elif i == 4:  # 9:34:00 - Consolidation
                current_price = 185.64
                open_price = 185.66
            elif i == 5:  # 9:35:00 - THE ENTRY CANDLE - closes above 3rd candle low
                current_price = 185.82  # Closes above 3rd candle low (trigger!)
                open_price = 185.66
            else:  # Continue movement
                trend = 0.002 if i < 60 else 0.001
                volatility = 0.03
                current_price += np.random.normal(trend, volatility)
                open_price = current_price + np.random.normal(0, 0.02)
                current_price = max(min(current_price, 190.00), 182.00)
            
            # Create OHLC
            if current_price >= open_price:
                high = current_price + abs(np.random.normal(0, 0.04))
                low = open_price - abs(np.random.normal(0, 0.03))
            else:
                high = open_price + abs(np.random.normal(0, 0.03))
                low = current_price - abs(np.random.normal(0, 0.04))
            
            prices.append({
                'Time': times[i],
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(current_price, 2),
                'Volume': np.random.randint(1000, 3000)
            })
        
        self.data_1m = pd.DataFrame(prices)

    def detect_fvg_and_entry(self):
        """Detect first FVG and entry signal"""
        if len(self.data_1m) < 3:
            return None
            
        # Check first 3 candles for FVG (indices 0, 1, 2)
        c1 = self.data_1m.iloc[0]
        c2 = self.data_1m.iloc[1]
        c3 = self.data_1m.iloc[2]
        
        # Check for Bullish FVG
        if c1['Low'] > c3['High']:
            fvg_type = "Bullish"
            fvg_open = c3['High']  # Bottom of gap
            fvg_close = c1['Low']  # Top of gap
            
            # Check if FVG aligns with daily bias
            if self.daily_bias == 'Bullish':
                self.first_fvg = {
                    'type': fvg_type,
                    'fvg_open': fvg_open,
                    'fvg_close': fvg_close,
                    'fvg_mid': (fvg_open + fvg_close) / 2,
                    'c1_low': c1['Low'],
                    'c3_low': c3['Low'],
                    'c3_high': c3['High'],
                    'candle_indices': [0, 1, 2],
                    'entry_trigger_level': c3['Low'],  # Must close above this
                    'entry_price': c3['Low'],  # Buy limit here
                    'stop_loss': c1['Low']  # Stop at 1st candle low
                }
                
                # Look for entry signal (candle closes above 3rd candle low)
                for i in range(3, min(len(self.data_1m), 90)):  # Check until 11:00 AM
                    candle = self.data_1m.iloc[i]
                    if candle['Close'] > c3['Low']:
                        self.entry_signal = {
                            'candle_index': i,
                            'entry_candle': candle,
                            'trigger_price': candle['Close'],
                            'entry_price': c3['Low'],
                            'stop_loss': c1['Low'],
                            'entry_time': candle['Time']
                        }
                        print(f"✅ ENTRY TRIGGERED!")
                        print(f"   Entry candle: {candle['Time'].strftime('%H:%M')} closes at ${candle['Close']:.2f}")
                        print(f"   Above trigger: ${c3['Low']:.2f}")
                        print(f"   Buy limit: ${c3['Low']:.2f}")
                        print(f"   Stop loss: ${c1['Low']:.2f}")
                        break
                
                return self.first_fvg
        
        # Check for Bearish FVG  
        elif c1['High'] < c3['Low']:
            fvg_type = "Bearish"
            fvg_open = c1['High']  # Top of gap
            fvg_close = c3['Low']  # Bottom of gap
            
            if self.daily_bias == 'Bearish':
                self.first_fvg = {
                    'type': fvg_type,
                    'fvg_open': fvg_open,
                    'fvg_close': fvg_close,
                    'fvg_mid': (fvg_open + fvg_close) / 2,
                    'c1_high': c1['High'],
                    'c3_high': c3['High'],
                    'c3_low': c3['Low'],
                    'candle_indices': [0, 1, 2],
                    'entry_trigger_level': c3['High'],  # Must close below this
                    'entry_price': c3['High'],  # Sell limit here
                    'stop_loss': c1['High']  # Stop at 1st candle high
                }
                
                # Look for entry signal (candle closes below 3rd candle high)
                for i in range(3, min(len(self.data_1m), 90)):
                    candle = self.data_1m.iloc[i]
                    if candle['Close'] < c3['High']:
                        self.entry_signal = {
                            'candle_index': i,
                            'entry_candle': candle,
                            'trigger_price': candle['Close'],
                            'entry_price': c3['High'],
                            'stop_loss': c1['High'],
                            'entry_time': candle['Time']
                        }
                        break
                
                return self.first_fvg
        
        return None

    def plot_complete_strategy(self):
        """Plot the complete strategy with FVG and entry signal"""
        if not self.first_fvg:
            print("No valid FVG found")
            return
            
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot first 60 minutes of data to focus on entry
        plot_data = self.data_1m.iloc[:60]
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            edge_color = 'darkgreen' if row['Close'] >= row['Open'] else 'darkred'
            
            # Highlight entry candle
            if self.entry_signal and idx == self.entry_signal['candle_index']:
                color = 'gold'
                edge_color = 'orange'
            
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Open'], row['Close'])
            
            ax.bar(row['Time'], body_height, bottom=body_bottom, 
                   width=timedelta(minutes=0.7), color=color, 
                   edgecolor=edge_color, alpha=0.8, linewidth=1)
            
            # Draw wicks
            ax.plot([row['Time'], row['Time']], [row['Low'], row['High']], 
                    color='black', linewidth=1, alpha=0.7)
        
        # Draw FVG zone
        fvg = self.first_fvg
        zone_start = self.data_1m.iloc[2]['Time']  # From 3rd candle
        zone_end = plot_data['Time'].iloc[-1]
        
        fvg_color = 'blue' if fvg['type'] == 'Bullish' else 'red'
        ax.axhspan(fvg['fvg_open'], fvg['fvg_close'], 
                   xmin=(zone_start - plot_data['Time'].iloc[0]).total_seconds() / (zone_end - plot_data['Time'].iloc[0]).total_seconds(),
                   alpha=0.25, color=fvg_color, zorder=1)
        
        # FVG boundary lines
        ax.axhline(fvg['fvg_open'], color=fvg_color, linestyle='-', linewidth=2, alpha=0.8)
        ax.axhline(fvg['fvg_close'], color=fvg_color, linestyle='-', linewidth=2, alpha=0.8)
        ax.axhline(fvg['fvg_mid'], color='purple', linestyle='--', linewidth=2, alpha=0.7)
        
        # Entry and stop loss lines
        if self.entry_signal:
            entry_price = self.entry_signal['entry_price']
            stop_loss = self.entry_signal['stop_loss']
            
            ax.axhline(entry_price, color='lime', linestyle='-', linewidth=3, alpha=0.9)
            ax.axhline(stop_loss, color='red', linestyle='-', linewidth=3, alpha=0.9)
            
            # Entry arrow
            entry_time = self.entry_signal['entry_time']
            entry_candle_close = self.entry_signal['trigger_price']
            
            if fvg['type'] == 'Bullish':
                ax.annotate('ENTRY', xy=(entry_time, entry_candle_close), 
                           xytext=(entry_time, entry_candle_close + 0.15),
                           ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color='white',
                           arrowprops=dict(arrowstyle='->', color='lime', lw=3),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.9))
        
        # Format chart
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.set_title(f'{fvg["type"]} FVG Entry Strategy - Daily Bias: {self.daily_bias}', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($/cwt)', fontsize=12)
        ax.set_xlabel('Time (EST)', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        # Set appropriate y-limits
        y_min = plot_data['Low'].min() - 0.2
        y_max = plot_data['High'].max() + 0.2
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        return fig

# Run the complete system
if __name__ == "__main__":
    print("🐄 Complete FVG Trading System with Entry/Exit Logic 🐄")
    print("=" * 60)
    
    # Initialize system
    system = CompleteFVGTradingSystem()
    
    # Generate daily data and determine bias
    system.generate_daily_data()
    
    # Generate 1-minute data with entry scenario
    system.create_fvg_with_entry_example()
    
    # Detect FVG and entry signal
    fvg = system.detect_fvg_and_entry()
    
    if fvg:
        print(f"\n🎯 FIRST FVG DETECTED:")
        print(f"Type: {fvg['type']} FVG")
        print(f"FVG Zone: ${fvg['fvg_open']:.2f} - ${fvg['fvg_close']:.2f}")
        print(f"Entry Price: ${fvg['entry_price']:.2f}")
        print(f"Stop Loss: ${fvg['stop_loss']:.2f}")
        
        if system.entry_signal:
            entry = system.entry_signal
            risk = abs(entry['entry_price'] - entry['stop_loss'])
            print(f"\n✅ ENTRY EXECUTED:")
            print(f"Entry Time: {entry['entry_time'].strftime('%H:%M:%S')}")
            print(f"Risk per contract: ${risk:.2f}")
    
    # Plot the strategy
    system.plot_complete_strategy()
    plt.show()
