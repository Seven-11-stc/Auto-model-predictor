# AutoML Model Recommender

An automated machine learning application that helps you find the best model for your dataset by comparing multiple algorithms and their performance metrics.

## Features

- 📊 Automated data preprocessing
- 🤖 Multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, KNN, Naive Bayes, Neural Networks)
- 📈 Comprehensive model evaluation (Accuracy, Precision, Recall, F1-Score, AUC)
- 🔍 Hyperparameter tuning using GridSearchCV
- 📊 Interactive visualizations
- 🎯 Best model recommendation

## Installation

1. Clone the repository:
```bash
git clone [YOUR_REPO_URL]
cd automl_model_selector
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset in the `dataset` folder (CSV format)
2. Run the application:
```bash
streamlit run app.py
```

3. In the web interface:
   - Upload your dataset
   - Click "Preprocess Data"
   - Click "Train and Evaluate Models"
   - View the results and download the best model

## Project Structure

```
automl_model_selector/
├── app.py                  # Main Streamlit application
├── preprocess.py           # Data preprocessing pipeline
├── model_selector.py       # Model training and evaluation
├── utils.py                # Utility functions and plotting
├── dataset/                # Input datasets
├── models/                 # Saved model files
├── results/                # Model evaluation results
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Requirements

- Python 3.7+
- Git
- A web browser

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
