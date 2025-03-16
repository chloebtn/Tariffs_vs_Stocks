import yfinance as yf
import pandas as pd

#Get data
nucor = yf.download('NUE', start='2024-01-01', end='2025-03-15')        # for steel
alcoa = yf.download('AA', start='2024-01-01', end='2025-03-15')         # for aluminum
ford = yf.download('F', start='2024-01-01', end='2025-03-15')
general_motors = yf.download('GM', start='2024-01-01', end='2025-03-15')
sp500 = yf.download('^GSPC', start='2024-01-01', end='2025-03-15')

# Returns
nucor_returns = nucor['Close'].pct_change().dropna()
alcoa_returns = alcoa['Close'].pct_change().dropna()
ford_returns = ford['Close'].pct_change().dropna()
gmotors_returns = general_motors['Close'].pct_change().dropna()
sp500_returns = sp500['Close'].pct_change().dropna()

# Summary statistics of returns
print("Nucor returns summary:", nucor_returns.describe())
print("\nAlcoa returns summary:", alcoa_returns.describe())
print("\nFord returns summary:", ford_returns.describe())
print("\nGeneral Motors returns summary:", gmotors_returns.describe())
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

tariff_announcement_date = '2025-02-01'
tariff_implementation_date = '2025-03-04'

#Stock performace before and after Tariff
# Nucor
nucor_before = nucor.loc[:tariff_announcement_date, 'Close'].mean()
nucor_between = nucor.loc[tariff_announcement_date:tariff_implementation_date, 'Close'].mean()
nucor_after = nucor.loc[tariff_implementation_date:, 'Close'].mean()

print(f"Nucor closing price before tariff: {nucor_before}, \nNucor closing price after tariff announcement: {nucor_between}")

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = nucor[['Open', 'High', 'Low']]
y = nucor['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model score:", model.score(X_test, y_test))

plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Closing Prices')
plt.plot(predictions, label='Predicted Closing Prices')
plt.title('Actual vs. Predicted Closing Prices')
plt.xlabel('Data Points')
plt.ylabel('Price')
plt.legend()
plt.show()