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
nucor_returns = nucor['Close'].pct_change().dropna()
alcoa_returns = alcoa['Close'].pct_change().dropna()
ford_returns = ford['Close'].pct_change().dropna()
g_motors_returns = general_motors['Close'].pct_change().dropna()
sp500_returns = sp500['Close'].pct_change().dropna()

# Summary statistics of returns
print("Nucor returns summary:", nucor_returns.describe())
print("\nAlcoa returns summary:", alcoa_returns.describe())
print("\nFord returns summary:", ford_returns.describe())
print("\nGeneral Motors returns summary:", g_motors_returns.describe())
print("\nS&P 500 returns summary:", sp500_returns.describe())

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(nucor['Close'], label='Nucor')
plt.plot(alcoa['Close'], label='Alcoa')
plt.plot(ford['Close'], label='Ford')
plt.plot(general_motors['Close'], label='General Motors')
plt.plot(sp500['Close'], label='S&P 500')
plt.title('Stock and Index Performance')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


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




# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = nucor[['Open', 'High', 'Low']].shift(1).dropna()
y = nucor.loc[X.index, 'Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel score:", model.score(X_test, y_test))

last_day_data = nucor[['Open', 'High', 'Low']].iloc[-1].values.reshape(1, -1)
next_day_prediction = model.predict(last_day_data)

print(f"\nPredicted closing price for the next day: {next_day_prediction[0]}")
