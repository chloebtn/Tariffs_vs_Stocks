import yfinance as yf
import pandas as pd

#Get data

start_date = '2024-07-01'
end_date = '2025-03-15'

nucor = yf.download('NUE', start=start_date, end=end_date)        
alcoa = yf.download('AA', start=start_date, end=end_date)         
ford = yf.download('F', start=start_date, end=end_date)
general_motors = yf.download('GM', start=start_date, end=end_date)
sp500 = yf.download('^GSPC', start=start_date, end=end_date)

nucor.index = pd.to_datetime(nucor.index)
alcoa.index = pd.to_datetime(alcoa.index)
ford.index = pd.to_datetime(ford.index)
general_motors.index = pd.to_datetime(general_motors.index)
sp500.index = pd.to_datetime(sp500.index)

# Returns
nucor['Returns'] = nucor['Close'].pct_change().dropna()
alcoa['Returns'] = alcoa['Close'].pct_change().dropna()
ford['Returns'] = ford['Close'].pct_change().dropna()
general_motors['Returns'] = general_motors['Close'].pct_change().dropna()
sp500['Returns'] = sp500['Close'].pct_change().dropna()

# Summary statistics of returns
print("Nucor returns summary:", nucor['Returns'].describe())
print("\nAlcoa returns summary:", alcoa['Returns'].describe())
print("\nFord returns summary:", ford['Returns'].describe())
print("\nGeneral Motors returns summary:", general_motors['Returns'].describe())
print("\nS&P 500 returns summary:", sp500['Returns'].describe())


# ANALYSIS

election_date = pd.to_datetime('2024-11-05')
tariff_announcement_date = pd.to_datetime('2025-02-01')
tariff_implementation_date = pd.to_datetime('2025-03-04')

stocks = {
    'Nucor': nucor,
    'Alcoa': alcoa,
    'Ford': ford,
    'General Motors': general_motors,
    'S&P 500': sp500
}

cumulative_returns = {
    'Nucor': (1 + nucor['Returns']).cumprod() - 1,
    'Alcoa': (1 + alcoa['Returns']).cumprod() - 1,
    'Ford': (1 + ford['Returns']).cumprod() - 1,
    'General Motors': (1 + general_motors['Returns']).cumprod() - 1,
    'S&P 500': (1 + sp500['Returns']).cumprod() - 1
}

import matplotlib.pyplot as plt

for name, data in cumulative_returns.items():
    plt.plot(data.index, data, label=name, linewidth=0.8)

plt.axvline(x=tariff_announcement_date, color='b', linestyle='--', linewidth=0.5, label='Tariff Announcement')
plt.axvline(x=tariff_implementation_date, color='g', linestyle='--', linewidth=0.5, label='Tariff Implementation')
plt.axvline(x=election_date, color='r', linestyle='--', linewidth=0.5, label='Election Date')
plt.title('Cumulative Returns of Stocks and Index')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Cumulative Return')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()


# Prices
for name, df in stocks.items():
    before = df.loc[:election_date, 'Close'].mean()
    between = df.loc[tariff_announcement_date:tariff_implementation_date, 'Close'].mean()
    after = df.loc[tariff_implementation_date:, 'Close'].mean()

    print(f"\n{name} closing price before election: {before.values}")
    print(f"{name} closing price after tariff announcement: {between.values}")
    print(f"{name} closing price after tariff implementation: {after.values}")


    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label=name)
    
    plt.axvline(x=tariff_announcement_date, color='b', linestyle='--', label='Tariff Announcement')
    plt.axvline(x=tariff_implementation_date, color='g', linestyle='--', label='Tariff Implementation')
    plt.axvline(x=election_date, color='r', linestyle='--', label='Election Date')

    plt.title(f'{name} Stock Price with Tariff Events')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

# Corelation Matrix
correlation_matrix = pd.DataFrame({
    'Nucor': nucor['Returns'],
    'Alcoa': alcoa['Returns'],
    'Ford': ford['Returns'],
    'General Motors': general_motors['Returns'],
    'S&P 500': sp500['Returns']
}).corr()

import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', square=True)
plt.title('Correlation Matrix')
plt.show()

#Volatility
for name, df in stocks.items():
    before_vol = df.loc[:election_date, 'Returns'].std()
    between_vol = df.loc[tariff_announcement_date:tariff_implementation_date, 'Returns'].std()
    after_vol = df.loc[tariff_implementation_date:, 'Returns'].std()

    print(f"\n{name} volatility before election: {before_vol}")
    print(f"{name} volatility after tariff announcement: {between_vol}")
    print(f"{name} volatility after tariff implementation: {after_vol}")


# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

for name, df in stocks.items():
    X = df[['Open', 'High', 'Low', 'Close']].shift(1).dropna()
    y = df.loc[X.index, 'Close']
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"\n{name} model score: {model.score(X_test, y_test)}")

    last_day_data = df[['Open', 'High', 'Low', 'Close']].iloc[-1].values.reshape(1, -1)
    next_day_prediction = model.predict(last_day_data)

    print(f"{name} predicted closing price for the next day: {next_day_prediction[0]}")
