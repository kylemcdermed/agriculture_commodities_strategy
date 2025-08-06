import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

class CornMarketTrendAnalyzer:
    def __init__(self, sma_period=7):
        """
        Initialize the Corn Market Trend Analyzer
        
        Args:
            sma_period (int): Period for Simple Moving Average calculation
        """
        self.sma_period = sma_period
        self.data = None
        self.signals = None
        
    def fetch_corn_data(self, symbol="CORN", period="1y"):
        """
        Fetch corn futures data
        
        Args:
            symbol (str): Trading symbol for corn futures
            period (str): Time period for data fetch
        """
        try:
            # Fetch corn futures data (using a corn ETF as proxy)
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(period=period)
            print(f"Successfully fetched {len(self.data)} days of {symbol} data")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Generate synthetic data for demonstration
            self.generate_synthetic_data()
            return False
    
    def generate_synthetic_data(self, days=365):
        """
        Generate synthetic corn price data for testing
        """
        print("Generating synthetic corn price data...")
        
        # Start date
        start_date = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        # Generate realistic corn price movement
        np.random.seed(42)
        base_price = 600  # Cents per bushel
        price_changes = np.random.normal(0, 15, days)  # Daily volatility
        trend = np.sin(np.arange(days) * 2 * np.pi / 250) * 50  # Seasonal trend
        
        prices = [base_price]
        for i in range(1, days):
            new_price = prices[-1] + price_changes[i] + trend[i] * 0.1
            prices.append(max(new_price, 300))  # Floor price
        
        self.data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(50000, 200000, days)
        }, index=dates)
        
        print(f"Generated {len(self.data)} days of synthetic data")
    
    def calculate_sma(self):
        """
        Calculate Simple Moving Average and trend signals
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        # Calculate SMA
        self.data[f'SMA_{self.sma_period}'] = self.data['Close'].rolling(
            window=self.sma_period
        ).mean()
        
        # Determine trend bias
        self.data['Trend_Bias'] = np.where(
            self.data['Close'] > self.data[f'SMA_{self.sma_period}'],
            'Bullish',
            'Bearish'
        )
        
        # Create signal changes
        self.data['Signal_Change'] = self.data['Trend_Bias'].ne(
            self.data['Trend_Bias'].shift()
        )
        
        print(f"Calculated {self.sma_period}-period SMA and trend signals")
    
    def get_current_bias(self):
        """
        Get the current market bias
        """
        if self.data is None or f'SMA_{self.sma_period}' not in self.data.columns:
            return None
        
        latest = self.data.iloc[-1]
        current_price = latest['Close']
        current_sma = latest[f'SMA_{self.sma_period}']
        current_bias = latest['Trend_Bias']
        
        return {
            'date': latest.name.strftime('%Y-%m-%d'),
            'close_price': round(current_price, 2),
            'sma_value': round(current_sma, 2),
            'bias': current_bias,
            'price_vs_sma': round(((current_price / current_sma) - 1) * 100, 2)
        }
    
    def get_signal_summary(self):
        """
        Get summary of trend signals
        """
        if self.data is None:
            return None
        
        # Remove NaN values
        clean_data = self.data.dropna()
        
        total_days = len(clean_data)
        bullish_days = len(clean_data[clean_data['Trend_Bias'] == 'Bullish'])
        bearish_days = len(clean_data[clean_data['Trend_Bias'] == 'Bearish'])
        
        # Signal changes
        signal_changes = clean_data['Signal_Change'].sum()
        
        return {
            'total_trading_days': total_days,
            'bullish_days': bullish_days,
            'bearish_days': bearish_days,
            'bullish_percentage': round((bullish_days / total_days) * 100, 1),
            'bearish_percentage': round((bearish_days / total_days) * 100, 1),
            'signal_changes': signal_changes
        }
    
    def plot_trend_analysis(self, days_to_show=90):
        """
        Plot the price data with SMA and trend signals
        """
        if self.data is None:
            raise ValueError("No data available")
        
        # Get recent data
        recent_data = self.data.tail(days_to_show)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Price and SMA plot
        ax1.plot(recent_data.index, recent_data['Close'], 
                label='Corn Close Price', linewidth=2, color='#2E8B57')
        ax1.plot(recent_data.index, recent_data[f'SMA_{self.sma_period}'], 
                label=f'{self.sma_period}-SMA', linewidth=2, color='#FF6B35')
        
        # Color background based on trend
        for i in range(len(recent_data) - 1):
            if recent_data.iloc[i]['Trend_Bias'] == 'Bullish':
                ax1.axvspan(recent_data.index[i], recent_data.index[i+1], 
                           alpha=0.1, color='green')
            else:
                ax1.axvspan(recent_data.index[i], recent_data.index[i+1], 
                           alpha=0.1, color='red')
        
        ax1.set_title('Corn Price Trend Analysis', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (cents/bushel)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Trend bias plot
        trend_numeric = recent_data['Trend_Bias'].map({'Bullish': 1, 'Bearish': -1})
        colors = ['red' if x == -1 else 'green' for x in trend_numeric]
        
        ax2.bar(recent_data.index, trend_numeric, color=colors, alpha=0.7, width=0.8)
        ax2.set_title('Daily Trend Bias', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Bias', fontsize=12)
        ax2.set_yticks([-1, 1])
        ax2.set_yticklabels(['Bearish', 'Bullish'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
        
        # Print current status
        current = self.get_current_bias()
        if current:
            print(f"\n🌽 CURRENT CORN MARKET STATUS 🌽")
            print(f"Date: {current['date']}")
            print(f"Close Price: ${current['close_price']:.2f}")
            print(f"7-SMA: ${current['sma_value']:.2f}")
            print(f"Current Bias: {current['bias']} ({'📈' if current['bias'] == 'Bullish' else '📉'})")
            print(f"Price vs SMA: {current['price_vs_sma']:+.2f}%")

# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = CornMarketTrendAnalyzer(sma_period=7)
    
    # Try to fetch real data, fallback to synthetic
    print("🌽 Corn Market Trend Analyzer 🌽")
    print("=" * 40)
    
    analyzer.fetch_corn_data()
    analyzer.calculate_sma()
    
    # Get summary
    summary = analyzer.get_signal_summary()
    print(f"\n📊 TREND ANALYSIS SUMMARY:")
    print(f"Total Trading Days: {summary['total_trading_days']}")
    print(f"Bullish Days: {summary['bullish_days']} ({summary['bullish_percentage']}%)")
    print(f"Bearish Days: {summary['bearish_days']} ({summary['bearish_percentage']}%)")
    print(f"Signal Changes: {summary['signal_changes']}")
    
    # Plot the analysis
    analyzer.plot_trend_analysis()
