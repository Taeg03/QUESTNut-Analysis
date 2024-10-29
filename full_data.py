import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the data
df_cleaned = pd.read_csv("data/all-deals.csv")

df_cleaned = df_cleaned.dropna(subset=["Amount"])
df_cleaned = df_cleaned.dropna(subset=["Create Date"])
df_cleaned = df_cleaned.dropna(subset=["Close Date"])

#print(df_cleaned)

df_cleaned["Create Date"] = pd.to_datetime(df_cleaned["Create Date"], format='mixed')
df_cleaned["Close Date"] = pd.to_datetime(df_cleaned["Close Date"], format='mixed')
df_cleaned["Time to Close"] = (df_cleaned["Close Date"] - df_cleaned["Create Date"]).dt.days


X = df_cleaned["Time to Close"]    # Independent variable
y = df_cleaned["Amount"]    # Dependent variable

X = sm.add_constant(X)
lm = sm.OLS(y, X).fit()
predicted_sales = lm.predict(X)


plt.scatter(df_cleaned['Time to Close'], df_cleaned['Amount'], label='Actual Sales', linewidth=3)
plt.plot(df_cleaned['Time to Close'], predicted_sales, label='MLR Model', color='orange', linewidth=3)

plt.xlabel('Time to Close')  # X-axis label
plt.ylabel('Amount')  # Y-axis label
plt.title('Sales over Time to Close')  # Plot title
plt.legend()
plt.show()