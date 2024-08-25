import pandas as pd

def category_combination(data, columns, threshold=5):
    # columns: desired category variables which you combine for stratify
    # threshold: When you split into train/test data. The test set requires a minimum of records.
    #            This threshold is that minimum value for the combination.

    # Combine the specified columns in a new column 'combination'
    data['combination'] = data[columns].astype(str).agg(''.join, axis=1)
    
    # Count the numbers of records by combination
    combination_values = data['combination'].value_counts()
    
    # Selecting the combination by the threshold
    valid_combination = combination_values[combination_values >= threshold].index
    
    # Data set ready to stratified split
    filtered_data = data[data['combination'].isin(valid_combination)]
    
    return filtered_data
