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
        """Generate daily data for 7-SMA trend bias.
        Note: Seed 150 produces a bullish bias (Close > SMA_7).
        To test a bearish bias, change the seed to 151 or adjust the trend logic.
        """
        dates = pd.date_range(start=datetime.now() - timedelta(days=40), end=datetime.now(), freq='D')
        
        np.random.seed(150)  # Seed for bullish bias; use 151 for bearish bias
        base_price = 185.00
        daily_prices = []
        
        for i, date in enumerate(dates):
            if i == 0:
                price = base_price
            else:
                trend = -0.025 if i < 20 else -0.015  # Slight downward trend
                price = daily_prices[-1] + np.random.normal(trend, 0.25)
                price = max(price, 175.00)
                price = min(price, 190.00)
            
            daily_prices.append(price)
            
        self.daily_data = pd.DataFrame({
            'Date': dates,
            'Open': [p + np.random.normal(0, 0.1) for p in daily_prices],
            'High': [p + abs(np.random.normal(0, 0.3)) for p in daily_prices],
            'Low': [p - abs(np.random.normal(0, 0.3)) for p in daily_prices],
            'Close': daily_prices
        })
        
        self.daily_data['SMA_7'] = self.daily_data['Close'].rolling(window=self.sma_period).mean()
        latest = self.daily_data.iloc[-1]
        self.daily_bias = 'Bullish' if latest['Close'] > latest['SMA_7'] else 'Bearish'
        
        print(f"Daily Bias: {self.daily_bias} (Close: ${latest['Close']:.2f} vs 7-SMA: ${latest['SMA_7']:.2f})")

    def create_fvg_with_entry_example(self):
        """Create 1-minute data to test for first valid FVG within 9:30-11:00 AM"""
        base_time = datetime(2024, 12, 15, 9, 30, 0)  # Start at 9:30 AM
        times = [base_time + timedelta(minutes=i) for i in range(120)]
        
        np.random.seed(750)  # Random seed for data generation
        prices = []
        overnight_close = 184.85
        current_price = overnight_close
        
        for i in range(120):
            trend = np.random.choice([-0.05, 0.05], p=[0.5, 0.5]) if i < 60 else np.random.choice([-0.01, 0.01], p=[0.5, 0.5])
            volatility = 0.05 if i < 60 else 0.025
            current_price += np.random.normal(trend, volatility)
            open_price = current_price + np.random.normal(0, 0.015)
            current_price = max(min(current_price, 186.00), 182.00)
            
            if current_price >= open_price:
                high = current_price + abs(np.random.normal(0, 0.03))
                low = open_price - abs(np.random.normal(0, 0.025))
            else:
                high = open_price + abs(np.random.normal(0, 0.025))
                low = current_price - abs(np.random.normal(0, 0.03))
            
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
        """Detect first FVG within 9:30-11:00 AM and check for entry based on daily bias and candle closes"""
        time_start = datetime(2024, 12, 15, 9, 30)
        time_end = datetime(2024, 12, 15, 11, 0)
        
        # Iterate through candles to find the first valid FVG
        for i in range(len(self.data_1m) - 2):
            c1 = self.data_1m.iloc[i]
            c2 = self.data_1m.iloc[i+1]
            c3 = self.data_1m.iloc[i+2]
            
            # Ensure third candle is within 9:30-11:00 AM
            if not (time_start <= c3['Time'] <= time_end):
                continue
                
            # Initialize FVG dictionary
            fvg = None
            
            # Check for FVG (c1['Low'] > c3['High'])
            if c1['Low'] > c3['High']:
                fvg_open = c3['High']
                fvg_close = c1['Low']
                fvg = {
                    'type': 'First FVG',
                    'fvg_open': fvg_open,
                    'fvg_close': fvg_close,
                    'fvg_mid': (fvg_open + fvg_close) / 2,
                    'c1_low': c1['Low'],
                    'c1_high': c1['High'],
                    'c3_low': c3['Low'],
                    'c3_high': c3['High'],
                    'candle_indices': [i, i+1, i+2],
                    'entry_trigger_level': c1['Low'] if self.daily_bias == 'Bullish' else c3['High'],
                }
                
            # Check for FVG (c1['High'] < c3['Low'])
            elif c1['High'] < c3['Low']:
                fvg_open = c1['High']
                fvg_close = c3['Low']
                fvg = {
                    'type': 'First FVG',
                    'fvg_open': fvg_open,
                    'fvg_close': fvg_close,
                    'fvg_mid': (fvg_open + fvg_close) / 2,
                    'c1_low': c1['Low'],
                    'c1_high': c1['High'],
                    'c3_low': c3['Low'],
                    'c3_high': c3['High'],
                    'candle_indices': [i, i+1, i+2],
                    'entry_trigger_level': c1['Low'] if self.daily_bias == 'Bullish' else c3['High'],
                }
                
            if fvg:
                self.first_fvg = fvg
                print(f"\n🎯 FIRST FVG DETECTED:")
                print(f"Type: {fvg['type']}")
                print(f"FVG Zone: ${fvg['fvg_open']:.2f} - ${fvg['fvg_close']:.2f}")
                
                # Check for entry based on daily bias
                if self.daily_bias == 'Bullish':
                    entry_price = c1['Low']  # Buy limit at 1st candle low (purple)
                    stop_loss = c3['Low']    # Stop at 3rd candle low (cyan)
                    risk = entry_price - stop_loss
                    take_profit = entry_price + risk
                    
                    self.first_fvg.update({
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk': risk,
                        'reward': risk
                    })
                    
                    # Look for entry signal: candle closing price greater than c1['Low']
                    for j in range(i+3, len(self.data_1m)):
                        candle = self.data_1m.iloc[j]
                        if time_start <= candle['Time'] <= time_end:
                            if candle['Close'] > c1['Low']:
                                self.entry_signal = {
                                    'candle_index': j,
                                    'entry_candle': candle,
                                    'trigger_price': candle['Close'],
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'risk': risk,
                                    'reward': risk,
                                    'entry_time': candle['Time']
                                }
                                print(f"✅ BULLISH ENTRY TRIGGERED!")
                                print(f"   Entry candle: {candle['Time'].strftime('%H:%M')} closing price ${candle['Close']:.2f} greater than ${c1['Low']:.2f}")
                                print(f"   Buy limit: ${entry_price:.2f}")
                                print(f"   Stop loss: ${stop_loss:.2f}")
                                print(f"   Take profit: ${take_profit:.2f}")
                                print(f"   Risk/Reward: ${risk:.2f} / ${risk:.2f} (1:1)")
                                return self.first_fvg
                    print(f"First FVG detected, no buy entry taken as no candle closing price was greater than ${c1['Low']:.2f} between 9:30-11:00 AM EST")
                
                elif self.daily_bias == 'Bearish':
                    entry_price = c3['High']  # Sell limit at 3rd candle high
                    stop_loss = c1['High']   # Stop at 1st candle high
                    risk = stop_loss - entry_price
                    take_profit = entry_price - risk
                    
                    self.first_fvg.update({
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk': risk,
                        'reward': risk
                    })
                    
                    # Look for entry signal: candle closing price less than c3['High']
                    for j in range(i+3, len(self.data_1m)):
                        candle = self.data_1m.iloc[j]
                        if time_start <= candle['Time'] <= time_end:
                            if candle['Close'] < c3['High']:
                                self.entry_signal = {
                                    'candle_index': j,
                                    'entry_candle': candle,
                                    'trigger_price': candle['Close'],
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'risk': risk,
                                    'reward': risk,
                                    'entry_time': candle['Time']
                                }
                                print(f"✅ BEARISH ENTRY TRIGGERED!")
                                print(f"   Entry candle: {candle['Time'].strftime('%H:%M')} closing price ${candle['Close']:.2f} less than ${c3['High']:.2f}")
                                print(f"   Sell limit: ${entry_price:.2f}")
                                print(f"   Stop loss: ${stop_loss:.2f}")
                                print(f"   Take profit: ${take_profit:.2f}")
                                print(f"   Risk/Reward: ${risk:.2f} / ${risk:.2f} (1:1)")
                                return self.first_fvg
                    print(f"First FVG detected, no sell entry taken as no candle closing price was less than ${c3['High']:.2f} between 9:30-11:00 AM EST")
                
                return self.first_fvg
        
        print("No valid FVG found between 9:30-11:00 AM EST")
        return None

    def plot_complete_strategy(self):
        """Plot the complete strategy with FVG and entry signal"""
        if not self.first_fvg:
            print("No valid FVG found between 9:30-11:00 AM EST")
            return
            
        fig, ax = plt.subplots(figsize=(16, 10))
        
        plot_data = self.data_1m.iloc[:60]
        
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            edge_color = 'darkgreen' if row['Close'] >= row['Open'] else 'darkred'
            
            if self.entry_signal and idx == self.entry_signal['candle_index']:
                color = 'gold'
                edge_color = 'orange'
            
            if i in self.first_fvg['candle_indices']:
                if i == self.first_fvg['candle_indices'][0]:
                    color = 'purple'
                    edge_color = 'darkviolet'
                elif i == self.first_fvg['candle_indices'][1]:
                    color = 'orange'
                    edge_color = 'darkorange'
                elif i == self.first_fvg['candle_indices'][2]:
                    color = 'cyan'
                    edge_color = 'darkcyan'
            
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Open'], row['Close'])
            
            ax.bar(row['Time'], body_height, bottom=body_bottom, 
                   width=timedelta(minutes=0.7), color=color, 
                   edgecolor=edge_color, alpha=0.8, linewidth=1)
            
            ax.plot([row['Time'], row['Time']], [row['Low'], row['High']], 
                    color='black', linewidth=1, alpha=0.7)
        
        c1_data = self.data_1m.iloc[self.first_fvg['candle_indices'][0]]
        c2_data = self.data_1m.iloc[self.first_fvg['candle_indices'][1]]
        c3_data = self.data_1m.iloc[self.first_fvg['candle_indices'][2]]
        
        ax.annotate('CANDLE #1\n(Purple)', xy=(c1_data['Time'], c1_data['High']), 
                    xytext=(c1_data['Time'], c1_data['High'] + 0.15),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.8))
        
        ax.annotate('CANDLE #2\n(Orange)', xy=(c2_data['Time'], c2_data['High']), 
                    xytext=(c2_data['Time'], c2_data['High'] + 0.15),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.8))
        
        ax.annotate('CANDLE #3\n(Cyan)', xy=(c3_data['Time'], c3_data['High']), 
                    xytext=(c3_data['Time'], c3_data['High'] + 0.15),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.8))
        
        fvg = self.first_fvg
        zone_start = self.data_1m.iloc[self.first_fvg['candle_indices'][2]]['Time']
        zone_end = plot_data['Time'].iloc[-1]
        
        # Color FVG zone blue if c3['High'] < c1['Low'], red otherwise
        fvg_color = 'blue' if fvg['fvg_open'] == c3['High'] else 'red'
        ax.axhspan(fvg['fvg_open'], fvg['fvg_close'], 
                   xmin=(zone_start - plot_data['Time'].iloc[0]).total_seconds() / (zone_end - plot_data['Time'].iloc[0]).total_seconds(),
                   alpha=0.25, color=fvg_color, zorder=1)
        
        ax.axhline(fvg['fvg_open'], color=fvg_color, linestyle='-', linewidth=2, alpha=0.8)
        ax.axhline(fvg['fvg_close'], color=fvg_color, linestyle='-', linewidth=2, alpha=0.8)
        ax.axhline(fvg['fvg_mid'], color='purple', linestyle='--', linewidth=2, alpha=0.7)
        
        if self.entry_signal:
            entry_price = self.entry_signal['entry_price']
            stop_loss = self.entry_signal['stop_loss']
            take_profit = self.entry_signal['take_profit']
            
            ax.axhline(entry_price, color='lime', linestyle='-', linewidth=3, alpha=0.9, label='Entry')
            ax.axhline(stop_loss, color='red', linestyle='-', linewidth=3, alpha=0.9, label='Stop Loss')
            ax.axhline(take_profit, color='blue', linestyle='-', linewidth=3, alpha=0.9, label='Take Profit')
            
            entry_time = self.entry_signal['entry_time']
            entry_candle_close = self.entry_signal['trigger_price']
            
            if self.daily_bias == 'Bullish':
                ax.annotate('ENTRY', xy=(entry_time, entry_candle_close), 
                           xytext=(entry_time, entry_candle_close + 0.15),
                           ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color='white',
                           arrowprops=dict(arrowstyle='->', color='lime', lw=3),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.9))
                
                ax.annotate('STOP', xy=(entry_time, stop_loss), 
                           xytext=(entry_time, stop_loss - 0.12),
                           ha='center', va='top', fontsize=10, fontweight='bold',
                           color='white',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.8))
            
            else:  # Bearish
                ax.annotate('ENTRY', xy=(entry_time, entry_candle_close), 
                           xytext=(entry_time, entry_candle_close - 0.15),
                           ha='center', va='top', fontsize=12, fontweight='bold',
                           color='white',
                           arrowprops=dict(arrowstyle='->', color='red', lw=3),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.9))
                
                ax.annotate('STOP', xy=(entry_time, stop_loss), 
                           xytext=(entry_time, stop_loss + 0.12),
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           color='white',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.8))
        
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.set_title(f'First FVG Strategy - 1:1 R/R - Daily Bias: {self.daily_bias}', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($/cwt)', fontsize=12)
        ax.set_xlabel('Time (EST)', fontsize=12)
        ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        y_min = plot_data['Low'].min() - 0.2
        y_max = plot_data['High'].max() + 0.2
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    print("🐄 Complete FVG Trading System - CANDLE IDENTIFICATION 🐄")
    print("=" * 60)
    print("📍 CANDLE IDENTIFICATION:")
    print("   CANDLE #1 = Purple = First FVG candle")
    print("   CANDLE #2 = Orange = Second FVG candle")
    print("   CANDLE #3 = Cyan = Third FVG candle")
    print("=" * 60)
    
    system = CompleteFVGTradingSystem()
    system.generate_daily_data()
    system.create_fvg_with_entry_example()
    fvg = system.detect_fvg_and_entry()
    
    if fvg and system.entry_signal:
        print(f"\n✅ COMPLETE TRADE SETUP:")
        entry = system.entry_signal
        print(f"Entry Time: {entry['entry_time'].strftime('%H:%M:%S')}")
        print(f"Entry: ${entry['entry_price']:.2f}")
        print(f"Stop: ${entry['stop_loss']:.2f}")
        print(f"TP: ${entry['take_profit']:.2f}")
        print(f"Risk: ${entry['risk']:.2f}")
        print(f"Reward: ${entry['reward']:.2f}")
        print(f"R/R Ratio: 1:1")
    
    system.plot_complete_strategy()
    plt.show()
