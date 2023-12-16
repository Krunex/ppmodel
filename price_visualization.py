import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Samsung Galaxy Tab Prices.xlsx'

# Load the Excel workbook
workbook = pd.ExcelFile(file_path)

# Extract sheet names
sheet_names = workbook.sheet_names

# Load data from each sheet and store in a dictionary
data_sheets = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names}

# Find common models across all sheets
common_models = set(data_sheets[sheet_names[0]]['Model'])
for sheet in sheet_names[1:]:
    common_models.intersection_update(set(data_sheets[sheet]['Model']))

# Aggregate data for common models
model_year_price = {model: {} for model in common_models}
for model in common_models:
    for sheet in sheet_names:
        year_data = data_sheets[sheet]
        if model in year_data['Model'].values:
            avg_price = year_data[year_data['Model'] == model]['Price ($)'].mean()
            model_year_price[model][int(sheet)] = avg_price

# Plotting
plt.figure(figsize=(10, 6))
for model, year_price in model_year_price.items():
    sorted_years = sorted(year_price.keys())
    prices = [year_price[year] for year in sorted_years]
    plt.plot(sorted_years, prices, label=model)

plt.xlabel('Year')
plt.ylabel('Average Price ($)')
plt.title('Price Trend of Common Models Over Years')
plt.legend()
plt.grid(True)
plt.show()
