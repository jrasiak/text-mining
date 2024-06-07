import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import unittest

warnings.filterwarnings('ignore')

nltk.download('wordnet')
nltk.download('stopwords')


class TextProcessor:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def normalize(self, text):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text

    def lemmatize(self, text):
        words = text.split()
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(lemmatized)


class SpamClassifier:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, encoding='latin1')
        self.processor = TextProcessor()
        self.model = RandomForestClassifier()

    def preprocess(self):
        self.data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        self.data.rename(columns={'v1': 'target', 'v2': 'Message'}, inplace=True)
        self.data['target'] = LabelEncoder().fit_transform(self.data['target'])
        self.data['Message'] = self.data['Message'].apply(self.processor.normalize).apply(self.processor.lemmatize)
        self.data = self.data.drop_duplicates(keep='first')

    def vectorize(self):
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        X = vectorizer.fit_transform(self.data['Message'])
        y = self.data['target']
        return X, y

    def resample(self, X, y):
        over = SMOTE(sampling_strategy=1)
        under = RandomUnderSampler(sampling_strategy=0.4)
        pipeline = Pipeline(steps=[('under', under), ('over', over)])
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        return X_resampled, y_resampled

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))
        print("ROC AUC Score:", roc_auc_score(y_test, predictions))

    def visualize(self):
        l = self.data['target'].value_counts()
        colors = ['#8BC34A', '#B2EBF2']

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
        ax[0].pie(l, explode=[0, 0.1], autopct='%1.1f%%', shadow=True, labels=['Ham', 'Spam'], colors=colors)
        ax[0].set_title('Target (%)')

        sns.countplot(x='target', data=self.data, palette=colors, edgecolor='black', ax=ax[1])
        ax[1].set_title('Number of Target')
        plt.show()


class TestSpamClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = SpamClassifier('spam.csv')
        self.classifier.preprocess()

    def test_preprocess(self):
        self.assertEqual(self.classifier.data.isnull().sum().sum(), 0)
        self.assertIn('Message', self.classifier.data.columns)

    def test_vectorize(self):
        X, y = self.classifier.vectorize()
        self.assertEqual(X.shape[0], y.shape[0])

    def test_resample(self):
        X, y = self.classifier.vectorize()
        X_resampled, y_resampled = self.classifier.resample(X, y)
        self.assertNotEqual(Counter(y_resampled), Counter(y))


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)

    classifier = SpamClassifier('spam.csv')
    classifier.preprocess()
    classifier.visualize()
    X, y = classifier.vectorize()
    X_resampled, y_resampled = classifier.resample(X, y)
    classifier.train(X_resampled, y_resampled)
