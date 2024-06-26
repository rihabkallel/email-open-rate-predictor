# analyse.py

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import regex as re
import pandas as pd


class Analyze:
    def plot_feature_analysis(self, data):

        # Creating new features
        data['length'] = data['subject'].apply(len)
        data['digit_count'] = data['subject'].apply(lambda x: len(re.findall(r'\d', x)))
        data['exclamation_count'] = data['subject'].apply(lambda x: x.count('！'))
        data['tone'] = data['tone']
        data['style'] = data['style']
        # date information (assuming you have a date in your dataset and the format is consistent)
        # data['date'] = data['subject'].apply(lambda x: re.findall(r'\d{4}/\d{1,2}/\d{1,2}', x))
        # data['date'] = data['date'].apply(lambda x: x[0] if x else '')


        plt.rcParams['figure.figsize'] = (10, 6)

        # Length vs Open Rate
        plt.figure()
        sns.boxplot(x='length', y='open_rate', data=data)
        plt.title('Length of Subject vs Open Rate')
        plt.xlabel('Length of Subject')
        plt.ylabel('Open Rate')
        plt.grid(True)
        plt.show()

        # Digit Count vs Open Rate
        plt.figure()
        sns.boxplot(x='digit_count', y='open_rate', data=data)
        plt.title('Digit Count vs Open Rate')
        plt.xlabel('Digit Count')
        plt.ylabel('Open Rate')
        plt.grid(True)
        plt.show()

        # Exclamation Marks Count vs Open Rate
        plt.figure()
        sns.boxplot(x='exclamation_count', y='open_rate', data=data)
        plt.title('Exclamation Marks Count vs Open Rate')
        plt.xlabel('Exclamation Marks Count')
        plt.ylabel('Open Rate')
        plt.grid(True)
        plt.show()
        
        # Language Style vs Open Rate
        plt.figure()
        sns.boxplot(x='style', y='open_rate', data=data)
        plt.title('Language Style vs Open Rate')
        plt.xlabel('Language Style')
        plt.ylabel('Open Rate')
        plt.grid(True)
        plt.show()
        
        # Tone vs Open Rate
        plt.figure()
        sns.boxplot(x='tone', y='open_rate', data=data)
        plt.title('Tone vs Open Rate')
        plt.xlabel('Tone')
        plt.ylabel('Open Rate')
        plt.grid(True)
        plt.show()

        # Date vs Open Rate
        # data['date'] = pd.to_datetime(data['date'], errors='coerce')
        # data['month_year'] = data['date'].dt.to_period('M')

        # plt.figure()
        # sns.boxplot(x='month_year', y='open_rate', data=data)
        # plt.title('Date vs Open Rate')
        # plt.xlabel('Date')
        # plt.ylabel('Open Rate')
        # plt.grid(True)
        # plt.xticks(rotation=45)
        # plt.show()



    def generate_wordcloud(self, text, font_path):
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Email Subjects')
        plt.show()
