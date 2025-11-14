def clean_data(data):
    """
    Cleans the input data by removing any NaN values and duplicates.
    """
    cleaned_data = data.dropna().drop_duplicates()
    return cleaned_data

def normalize_data(data):
    """
    Normalizes the input data to a range of 0 to 1.
    """
    return (data - data.min()) / (data.max() - data.min())

def split_data(data, train_size=0.8):
    """
    Splits the data into training and testing sets.
    
    Parameters:
    - data: The dataset to split.
    - train_size: The proportion of the dataset to include in the train split.
    
    Returns:
    - train_data: The training data.
    - test_data: The testing data.
    """
    train_data = data[:int(len(data) * train_size)]
    test_data = data[int(len(data) * train_size):]
    return train_data, test_data

def prepare_features_and_labels(data, label_column):
    """
    Prepares features and labels for model training.
    
    Parameters:
    - data: The dataset containing features and labels.
    - label_column: The name of the column to be used as labels.
    
    Returns:
    - features: The feature set.
    - labels: The labels.
    """
    features = data.drop(columns=[label_column])
    labels = data[label_column]
    return features, labels