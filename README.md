cool stock predictor
## Overview
Stock Predictor is a machine learning project designed to analyze historical stock market data and predict future price movements. This tool leverages data processing libraries and predictive modeling to provide insights into market trends.

## Project Structure
The repository contains the following core components:

*   **`main.py`**: The primary entry point for the application. It allows for data fetching, model training, and prediction output.
*   **`data_loader.py`**: Handles the retrieval of historical stock data using APIs (such as Yahoo Finance) and manages data cleaning and preprocessing.
*   **`model.py`**: Contains the architecture for the prediction models (e.g., LSTM, Random Forest, or Linear Regression).
*   **`utils.py`**: Helper functions for data visualization, technical indicator calculations (RSI, Moving Averages), and evaluation metrics.
*   **`requirements.txt`**: Lists all necessary Python dependencies required to run the project.

## Getting Started

### Prerequisites
- Python 3.8+
- Pip package manager

### Installation
1. Fork the repository.
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-predictor.git
   cd stock-predictor
   ```

   If you don't want to fork the repository, then enter this:
   ```bash
   git clone https://github.com/z8kee/stock-predictor.git
   cd stock-predictor\src
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Create a `.env` file and store OPENAI_API_KEY=sk... in it, or manually set it when running `python main.py`

Run the main script to start the prediction process:
```bash
python main.py
```

## Features
- **Historical Data Analysis**: Fetch and visualize years of stock history.
- **Technical Indicators**: Automatically calculate market indicators to improve model accuracy.
- **Future Forecasting**: Predict closing prices for the upcoming trading days.
- **Performance Evaluation**: View Mean Squared Error (MSE) and accuracy graphs for model validation.

This project is for educational purposes.
