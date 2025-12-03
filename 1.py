import pandas as pd

#Load the data file
df = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

# Initial inspection of the first 5 rows
print("Data Head:")
print(df.head())

# Check column names, data types, and null values
print("\nData Info:")
print(df.info())

#After loading the data we now will do cleaning
# 1. Rename and clean columns
new_columns = [
    'State', 'Date', 'Frequency', 'Unemployment Rate', 'Employed',
    'Labour Participation Rate', 'Region', 'Longitude', 'Latitude'
]
df.columns = new_columns

# 2. Convert 'Date' to datetime objects
# The 'dayfirst=True' argument is used because the dates are in DD-MM-YYYY format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# 3. Feature Engineering: Extract Month for seasonal analysis
df['Month'] = df['Date'].dt.strftime('%b') # Abbreviated month name (Jan, Feb, etc.)
df['Month_int'] = df['Date'].dt.month      # Month number (1, 2, 3, etc. for sorting)

# Verify the cleaned data
print("Cleaned Data Head:")
print(df.head())
print("\nCleaned Data Info:")
print(df.info())

#impact of covid19 on employment rates
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Calculate the national average unemployment rate per month
monthly_avg_unemployment = df.groupby('Date')['Unemployment Rate'].mean().reset_index()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg_unemployment['Date'], monthly_avg_unemployment['Unemployment Rate'], marker='o')

# Mark the start of the national lockdown (April 2020)
lockdown_start = pd.to_datetime('2020-04-01')
plt.axvline(x=lockdown_start, color='red', linestyle='--', label='National Lockdown Start')

plt.title('Monthly Average Unemployment Rate in India')
plt.xlabel('Date')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('monthly_unemployment_trend.png')
plt.close()

#  STATE-WISE IMPACT ANALYSIS

# Define time periods
# Pre-Lockdown: Jan 2020 to March 2020 (Months 1 to 3)
pre_lockdown = df[(df['Date'].dt.month >= 1) & (df['Date'].dt.month <= 3) & (df['Date'].dt.year == 2020)]

# Lockdown Peak: April 2020 to May 2020 (Months 4 to 5)
lockdown_peak = df[(df['Date'].dt.month >= 4) & (df['Date'].dt.month <= 5) & (df['Date'].dt.year == 2020)]

# 1. Calculate mean unemployment rates for each state in each period
pre_lockdown_ur = pre_lockdown.groupby('State')['Unemployment Rate'].mean().rename('Pre_Lockdown_UR')
lockdown_peak_ur = lockdown_peak.groupby('State')['Unemployment Rate'].mean().rename('Lockdown_Peak_UR')

# 2. Merge the results and calculate the change
ur_comparison = pd.concat([pre_lockdown_ur, lockdown_peak_ur], axis=1).reset_index()
ur_comparison['Change'] = ur_comparison['Lockdown_Peak_UR'] - ur_comparison['Pre_Lockdown_UR']

# 3. Sort by the magnitude of change (descending)
ur_comparison = ur_comparison.sort_values(by='Change', ascending=False)

# 4. Plot the change using a horizontal bar chart
plt.figure(figsize=(12, 10))
sns.barplot(data=ur_comparison, y='State', x='Change', hue='State', palette='viridis', legend=False)
plt.title('Change in Average Unemployment Rate (Lockdown Peak vs. Pre-Lockdown)', fontsize=16)
plt.xlabel('Increase in Unemployment Rate (%)', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.tight_layout()
plt.savefig('state_wise_unemployment_change.png')
plt.show()

#  REGIONAL PATTERN ANALYSIS

# 1. Calculate the average unemployment rate by geographical Region over the whole period
regional_avg_ur = df.groupby('Region')['Unemployment Rate'].mean().reset_index()

# 2. Sort by unemployment rate (descending)
regional_avg_ur = regional_avg_ur.sort_values(by='Unemployment Rate', ascending=False)

# 3. Plot the regional comparison
plt.figure(figsize=(10, 6))
# Use the fixed seaborn syntax here as well
sns.barplot(
    data=regional_avg_ur,
    x='Region',
    y='Unemployment Rate',
    hue='Region',  # Fixed for future compatibility
    palette='coolwarm',
    legend=False
)
plt.title('Average Unemployment Rate by Geographical Region (Jan - Nov 2020)', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Average Unemployment Rate (%)', fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig('regional_unemployment_comparison.png')
plt.show() # Display the plot