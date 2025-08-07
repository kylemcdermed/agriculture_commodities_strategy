import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

class CompleteFVGTradingSystem:
    def __init__(self, sma_period=7, account_balance=10000, win_prob=0.5):
        """Complete FVG Trading System with Single Trade Per Day and Kelly Criterion"""
        self.sma_period = sma_period
        self.account_balance = account_balance
        self.win_prob = win_prob  # Probability of winning (adjust based on data)
        self.data_1m = None
        self.daily_data = None
        self.daily_bias = None
        self.first_fvg = None
        self.entry_signal = None
        self.orders = []  # List to track active orders
        self.trade_taken = False  # Flag to ensure only one trade per day

    def generate_daily_data(self):
        """Generate daily data for 7-SMA trend bias"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=40), end=datetime.now(), freq='D')
        np.random.seed(150)  # Seed for bullish bias; use 151 for bearish
        base_price = 185.00
        daily_prices = []
        for i, date in enumerate(dates):
            if i == 0:
                price = base_price
            else:
                trend = -0.025 if i < 20 else -0.015
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
        base_time = datetime(2024, 12, 15, 9, 30, 0)
        times = [base_time + timedelta(minutes=i) for i in range(120)]
        np.random.seed(750)
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

    def calculate_kelly_fraction(self, reward_to_risk):
        """Calculate Kelly Criterion fraction for optimal position sizing"""
        p = self.win_prob
        q = 1 - p
        b = reward_to_risk
        f = (p * (b + 1) - q) / b if b > 0 else 0
        return max(0, min(0.5, f))  # Cap at 50% to avoid over-leveraging

    def detect_fvg_and_entry(self):
        """Detect first FVG and manage one trade per day with three orders"""
        if self.trade_taken:
            print("Trade already taken today, no further trades allowed.")
            return self.first_fvg
        time_start = datetime(2024, 12, 15, 9, 30)
        time_end = datetime(2024, 12, 15, 11, 0)
        for i in range(len(self.data_1m) - 2):
            c1 = self.data_1m.iloc[i]
            c2 = self.data_1m.iloc[i+1]
            c3 = self.data_1m.iloc[i+2]
            if not (time_start <= c3['Time'] <= time_end):
                continue
            fvg = None
            if c1['Low'] > c3['High']:
                fvg_open, fvg_close = c3['High'], c1['Low']
                fvg = {'type': 'First FVG', 'fvg_open': fvg_open, 'fvg_close': fvg_close, 'c1_low': c1['Low'],
                       'c1_high': c1['High'], 'c3_low': c3['Low'], 'c3_high': c3['High'], 'candle_indices': [i, i+1, i+2]}
            elif c1['High'] < c3['Low']:
                fvg_open, fvg_close = c1['High'], c3['Low']
                fvg = {'type': 'First FVG', 'fvg_open': fvg_open, 'fvg_close': fvg_close, 'c1_low': c1['Low'],
                       'c1_high': c1['High'], 'c3_low': c3['Low'], 'c3_high': c3['High'], 'candle_indices': [i, i+1, i+2]}
            if fvg:
                self.first_fvg = fvg
                print(f"\n🎯 FIRST FVG DETECTED:")
                print(f"Type: {fvg['type']}")
                print(f"FVG Zone: ${fvg['fvg_open']:.2f} - ${fvg['fvg_close']:.2f}")
                if self.daily_bias == 'Bullish':
                    entry_price = c1['Low']
                    stop_loss = c3['Low']
                    risk = entry_price - stop_loss
                    tp_1 = entry_price + risk  # 1:1 RR
                    tp_2 = entry_price + 2 * risk  # 1:2 RR
                    tp_3 = entry_price + 3 * risk  # 1:3 RR
                    kelly_fractions = [self.calculate_kelly_fraction(r) for r in [1, 2, 3]]
                    self.orders = [
                        {'rr': 1, 'entry': entry_price, 'stop': stop_loss, 'tp': tp_1, 'kelly': kelly_fractions[0]},
                        {'rr': 2, 'entry': entry_price, 'stop': stop_loss, 'tp': tp_2, 'kelly': kelly_fractions[1]},
                        {'rr': 3, 'entry': entry_price, 'stop': stop_loss, 'tp': tp_3, 'kelly': kelly_fractions[2]}
                    ]
                    for j in range(i+3, len(self.data_1m)):
                        candle = self.data_1m.iloc[j]
                        if time_start <= candle['Time'] <= time_end and candle['Close'] > entry_price and not self.trade_taken:
                            self.entry_signal = {'candle_index': j, 'entry_candle': candle, 'entry_time': candle['Time']}
                            self.trade_taken = True
                            print(f"✅ BULLISH ENTRY TRIGGERED at {candle['Time'].strftime('%H:%M')} (Close: ${candle['Close']:.2f} > ${entry_price:.2f})")
                            active_orders = self.orders.copy()
                            for order in active_orders:
                                print(f"   Order {order['rr']}:1 RR - Entry: ${order['entry']:.2f}, Stop: ${order['stop']:.2f}, TP: ${order['tp']:.2f}, Kelly: {order['kelly']:.2%}")
                                if candle['Close'] >= order['tp']:
                                    if order['rr'] == 1:
                                        for o in active_orders:
                                            o['stop'] = o['entry']
                                        print(f"   1:1 RR hit, stopped all to entry: ${entry_price:.2f}")
                                    elif order['rr'] == 2:
                                        for o in [o for o in active_orders if o['rr'] > 1]:
                                            o['stop'] = tp_1
                                        print(f"   1:2 RR hit, stopped remaining to 1:1 TP: ${tp_1:.2f}")
                                    elif order['rr'] == 3:
                                        active_orders = []
                                        print(f"   1:3 RR hit, all orders booked")
                            self.orders = active_orders
                            if not self.orders:
                                break
                elif self.daily_bias == 'Bearish':
                    entry_price = c3['High']
                    stop_loss = c1['High']
                    risk = stop_loss - entry_price
                    tp_1 = entry_price - risk  # 1:1 RR
                    tp_2 = entry_price - 2 * risk  # 1:2 RR
                    tp_3 = entry_price - 3 * risk  # 1:3 RR
                    kelly_fractions = [self.calculate_kelly_fraction(r) for r in [1, 2, 3]]
                    self.orders = [
                        {'rr': 1, 'entry': entry_price, 'stop': stop_loss, 'tp': tp_1, 'kelly': kelly_fractions[0]},
                        {'rr': 2, 'entry': entry_price, 'stop': stop_loss, 'tp': tp_2, 'kelly': kelly_fractions[1]},
                        {'rr': 3, 'entry': entry_price, 'stop': stop_loss, 'tp': tp_3, 'kelly': kelly_fractions[2]}
                    ]
                    for j in range(i+3, len(self.data_1m)):
                        candle = self.data_1m.iloc[j]
                        if time_start <= candle['Time'] <= time_end and candle['Close'] < entry_price and not self.trade_taken:
                            self.entry_signal = {'candle_index': j, 'entry_candle': candle, 'entry_time': candle['Time']}
                            self.trade_taken = True
                            print(f"✅ BEARISH ENTRY TRIGGERED at {candle['Time'].strftime('%H:%M')} (Close: ${candle['Close']:.2f} < ${entry_price:.2f})")
                            active_orders = self.orders.copy()
                            for order in active_orders:
                                print(f"   Order {order['rr']}:1 RR - Entry: ${order['entry']:.2f}, Stop: ${order['stop']:.2f}, TP: ${order['tp']:.2f}, Kelly: {order['kelly']:.2%}")
                                if candle['Close'] <= order['tp']:
                                    if order['rr'] == 1:
                                        for o in active_orders:
                                            o['stop'] = o['entry']
                                        print(f"   1:1 RR hit, stopped all to entry: ${entry_price:.2f}")
                                    elif order['rr'] == 2:
                                        for o in [o for o in active_orders if o['rr'] > 1]:
                                            o['stop'] = tp_1
                                        print(f"   1:2 RR hit, stopped remaining to 1:1 TP: ${tp_1:.2f}")
                                    elif order['rr'] == 3:
                                        active_orders = []
                                        print(f"   1:3 RR hit, all orders booked")
                            self.orders = active_orders
                            if not self.orders:
                                break
                return self.first_fvg
        print("No valid FVG found between 9:30-11:00 AM EST")
        return None

    def plot_complete_strategy(self):
        """Plot the complete strategy with FVG and single trade per day"""
        if not self.first_fvg or not self.entry_signal:
            print("No valid FVG or entry signal found")
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
            ax.bar(row['Time'], body_height, bottom=body_bottom, width=timedelta(minutes=0.7), color=color,
                   edgecolor=edge_color, alpha=0.8, linewidth=1)
            ax.plot([row['Time'], row['Time']], [row['Low'], row['High']], color='black', linewidth=1, alpha=0.7)
        c1_data, c2_data, c3_data = [self.data_1m.iloc[i] for i in self.first_fvg['candle_indices']]
        for c, label, color in [(c1_data, 'CANDLE #1\n(Purple)', 'purple'), (c2_data, 'CANDLE #2\n(Orange)', 'orange'),
                                (c3_data, 'CANDLE #3\n(Cyan)', 'cyan')]:
            ax.annotate(label, xy=(c['Time'], c['High']), xytext=(c['Time'], c['High'] + 0.15),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        fvg = self.first_fvg
        zone_start = c3_data['Time']
        zone_end = plot_data['Time'].iloc[-1]
        fvg_color = 'blue' if fvg['fvg_open'] == c3_data['High'] else 'red'
        ax.axhspan(fvg['fvg_open'], fvg['fvg_close'], xmin=(zone_start - plot_data['Time'].iloc[0]).total_seconds() /
                   (zone_end - plot_data['Time'].iloc[0]).total_seconds(), alpha=0.25, color=fvg_color, zorder=1)
        ax.axhline(fvg['fvg_open'], color=fvg_color, linestyle='-', linewidth=2, alpha=0.8)
        ax.axhline(fvg['fvg_close'], color=fvg_color, linestyle='-', linewidth=2, alpha=0.8)
        ax.axhline((fvg['fvg_open'] + fvg['fvg_close']) / 2, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        if self.orders:
            entry_price = self.orders[0]['entry']
            ax.axhline(entry_price, color='green', linestyle='-', linewidth=3, alpha=0.9, label='Entry')
            for order in self.orders:
                ax.axhline(order['stop'], color='red', linestyle='-', linewidth=3, alpha=0.9, label='Stop Loss')
                ax.axhline(order['tp'], color='blue', linestyle='--', linewidth=2, alpha=0.7,
                           label=f'TP {order['rr']}:1' if order['rr'] == 1 else "")
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.set_title(f'First FVG Strategy - 1:1, 1:2, 1:3 RR - Daily Bias: {self.daily_bias}', fontsize=16, fontweight='bold')
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
    if fvg and system.entry_signal and system.orders:
        print(f"\n✅ ACTIVE ORDERS:")
        for order in system.orders:
            print(f"Order {order['rr']}:1 RR - Entry: ${order['entry']:.2f}, Stop: ${order['stop']:.2f}, TP: ${order['tp']:.2f}, Kelly: {order['kelly']:.2%}")
    system.plot_complete_strategy()
    plt.show()
