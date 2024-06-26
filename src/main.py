# main.py

from preprocess import Preprocess
from analyze import Analyze
from train import ModelTrainer
from predict import Predictor
import pandas as pd

# Main execution starts here
if __name__ == "__main__":
    # Importing the dataset
    data = pd.read_csv('dataset/open_rate_dataset_en.csv')
    
    # Initialize Preprocessor and preprocess the data
    preprocessor = Preprocess()
    data = preprocessor.apply_preprocessing(data)
    
    print(data.head(2))

    # Initialize Analyzer and perform analysis
    analyzer = Analyze()
    analyzer.plot_feature_analysis(data)
    text = " ".join(subject for subject in data['clean_subject'])
    analyzer.generate_wordcloud(text, font_path='NotoSansJP-VF.ttf')

    # Initialize ModelTrainer and train the model
    trainer = ModelTrainer()
    pipeline, mse, rmse, r2, mape = trainer.train_model(data)

    # Print model evaluation metrics
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R^2 Score: {r2}')
    # print(f'MAPE: {mape * 100:.2f}%')

    # # Perform cross-validation
    # cross_val_scores = trainer.cross_validate_model(data)
    # print(f'Cross-Validation R^2 Scores: {cross_val_scores}')
    # print(f'Average Cross-Validation R^2 Score: {cross_val_scores.mean()}')

    # Initialize Predictor with the saved model
    predictor = Predictor('model/open_rate_predictor.pkl', preprocessor)

    # Predict open rate for a new subject
    new_subject = 'Unlock Your Potential with Our Latest Course!'
    # TODO instead, use Generative AI to analyse the tone and language style of the subject.
    style = 'confident'
    tone = 'persuasive'
    predicted_open_rate = predictor.predict_open_rate(new_subject, style, tone)
    print(f'Predicted Open Rate for "{new_subject}": {predicted_open_rate:.2f}')
