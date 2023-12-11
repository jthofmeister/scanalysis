import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP

def load_database(file_path):
    return pd.read_csv(file_path, engine='python')

def clean_data(data):
    return data.dropna()

def calculate_correlation(col1, col2):
    if len(col1.unique()) < 2 or len(col2.unique()) < 2: # Check if there are at least 2 unique values in each column, turn up if needed
        return 0, Decimal('0')
    correlation_coefficient, p_value = pearsonr(col1, col2)
    if p_value < 0.05: # Correlations with a higher p value than this will not be considered, change this value if loosening tolerance on significance
        p_value = Decimal(str(p_value)).quantize(Decimal('1e-5'), rounding=ROUND_HALF_UP)  
        return correlation_coefficient, p_value
    else:
        return 0, Decimal('0')

def find_all_correlations(data):
    numeric_columns = data.select_dtypes(include='number').columns
    all_correlations = []
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            col1 = data[numeric_columns[i]]
            col2 = data[numeric_columns[j]]
            correlation, p_value = calculate_correlation(col1, col2)
            all_correlations.append(((numeric_columns[i], numeric_columns[j]), correlation, p_value))
    all_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    return all_correlations

def plot_correlation(data, selected_cols, plot_number):
    plt.figure(plot_number)
    correlation_coefficient, p_value = pearsonr(data[selected_cols[0]], data[selected_cols[1]])
    slope, intercept = np.polyfit(data[selected_cols[0]], data[selected_cols[1]], 1)
    line_of_best_fit = [slope * i + intercept for i in data[selected_cols[0]]]
            
    if len(data) >= 5: # Checks to see if there are at least x number of unique data points per column, "If n < 30 we have to be careful about generalizing the data, because the population is not normal."
        if p_value < 0.05 and correlation_coefficient != 0: # Correlations with a higher p value than this will not be plotted, change this value if loosening tolerance on significance, set limit is academic gold standard in most scenarios
            plt.scatter(data[selected_cols[0]], data[selected_cols[1]])
            plt.xlabel(f'{selected_cols[0]}')
            plt.ylabel(f'{selected_cols[1]}')
            plt.title(f'Scatter Plot - Correlation {plot_number}\nP-value: {p_value:.5f}, Correlation Coefficient: {correlation_coefficient:.5f}\nEquation: y = {slope:.5f}x + {intercept:.5f}')
            plt.plot(data[selected_cols[0]], line_of_best_fit, color='gold', label='Line of Best Fit')
            print(f"\nData points for correlation between {selected_cols[0]} and {selected_cols[1]}:")
            print(f'P-value: {p_value:.5f}, Correlation Coefficient: {correlation_coefficient:.5f}')
            print(data[[selected_cols[0], selected_cols[1]]].head(25)) # Prints first 25 data points from each correlation that gets plotted
            plt.legend()
    else:
        print(f"Insufficient data points for correlation between {selected_cols[0]} and {selected_cols[1]}")

def main():
    file_path = input("Enter the path to your database file: ")
    data = load_database(file_path)
    cleaned_data = clean_data(data).dropna()
    p_value_threshold = 0.05 # This value can be decreased for more sensitivity, to turn up see 'def calculate_correlation' and also increase here
    all_correlations = find_all_correlations(cleaned_data)

    if not all_correlations:
        print("No correlations found.")
    else:
        top_x_correlations = []
        num_plots_generated = 0
        for correlation_info in all_correlations:
            selected_cols = correlation_info[0]
            correlation_coefficient, p_value = calculate_correlation(cleaned_data[selected_cols[0]], cleaned_data[selected_cols[1]])
            if p_value < p_value_threshold and correlation_coefficient != 0:
                top_x_correlations.append(correlation_info)
                num_plots_generated += 1
                plot_correlation(cleaned_data, selected_cols, num_plots_generated)
                if num_plots_generated == 3: # Change this value to change the number of graphs generated from the descending list of statistically significant correlations found in the dataset
                    break
        plt.show()
        
if __name__ == "__main__":
    main()
