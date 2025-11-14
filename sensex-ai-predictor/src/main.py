import requests
from src.data.fetch_indices import fetch_latest_indices
from src.models.predictor import Predictor

def main():
    # Fetch the latest Sensex indices and their market values
    indices = fetch_latest_indices()
    
    if not indices:
        print("No indices fetched. Exiting.")
        return

    # Initialize the predictor model
    predictor = Predictor()

    # Train the model with historical data (this should be implemented in the Predictor class)
    predictor.train(indices)

    # Predict the performance of shares for the next week
    predictions = predictor.predict_next_week()

    # Display the predicted share performance
    print("Predicted share performance for the next week:")
    for stock, prediction in predictions.items():
        print(f"{stock}: Predicted Value - {prediction}")

if __name__ == "__main__":
    main()