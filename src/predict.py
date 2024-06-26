# predict.py

import joblib
import pandas as pd

class Predictor:
    def __init__(self, model_path, preprocessor):
        self.pipeline = joblib.load(model_path)
        self.preprocessor = preprocessor

    def predict_open_rate(self, subject, style, tone):
        clean_subject = self.preprocessor.preprocess_text(subject)
        #TODO - Add more tone and style reading features
        data = pd.DataFrame({'clean_subject': [clean_subject], 'length': [len(subject)], 'digit_count': [sum(c.isdigit() for c in subject)], 'exclamation_count': [subject.count('!')], 'style': [style], 'tone': [tone]})
        predicted_open_rate = self.pipeline.predict(data)
        return predicted_open_rate[0]
