# Spam Email Classifier using Machine Learning
# Description: Detects and filters spam emails using Naive Bayes and SVM

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SpamEmailClassifier:
    def __init__(self):
        self.vectorizer = None
        self.naive_bayes_model = None
        self.svm_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_sample_data(self):
        spam_emails = [
            'Click here to win $1000 prize NOW!!!',
            'Limited time offer - buy 1 get 1 free',
            'Congratulations! You have won the lottery',
            'Act now! This offer expires in 24 hours',
            'Free money! No work required',
            'Update your account information URGENTLY',
            'You have been selected for a special offer',
            'Increase your earnings instantly',
            'viagra cialis prescription cheap',
            'Meet singles in your area'
        ]
        
        legitimate_emails = [
            'Meeting scheduled for tomorrow at 10 AM',
            'Project update: Phase 1 complete',
            'Your monthly report is ready for review',
            'Team lunch this Friday at noon',
            'Conference details and registration',
            'Welcome to our organization',
            'Quarterly earnings announcement',
            'Technical documentation for API',
            'Employee training schedule updated',
            'Password reset instructions'
        ]
        
        emails = spam_emails + legitimate_emails
        labels = [1] * len(spam_emails) + [0] * len(legitimate_emails)
        return emails, labels
    
    def prepare_data(self, emails, labels):
        print('[1] Preparing data with TF-IDF vectorization...')
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(emails)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        print(f'  Training: {len(self.y_train)}, Test: {len(self.y_test)}')
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_naive_bayes(self):
        print('[2] Training Naive Bayes classifier...')
        self.naive_bayes_model = MultinomialNB()
        self.naive_bayes_model.fit(self.X_train, self.y_train)
        print('  Naive Bayes trained!')
    
    def train_svm(self):
        print('[3] Training SVM classifier...')
        self.svm_model = LinearSVC(random_state=42, max_iter=2000)
        self.svm_model.fit(self.X_train, self.y_train)
        print('  SVM trained!')
    
    def evaluate_model(self, model, model_name):
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        print(f'{model_name} Metrics:')
        print(f'  Accuracy: {accuracy:.2%}')
        print(f'  Precision: {precision:.2%}')
        print(f'  Recall: {recall:.2%}')
        print(f'  F1-Score: {f1:.2%}')
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def classify_email(self, email_text, model_type='naive_bayes'):
        email_vector = self.vectorizer.transform([email_text])
        
        if model_type == 'naive_bayes':
            prediction = self.naive_bayes_model.predict(email_vector)[0]
            confidence = max(self.naive_bayes_model.predict_proba(email_vector)[0])
        else:
            prediction = self.svm_model.predict(email_vector)[0]
            confidence = abs(self.svm_model.decision_function(email_vector)[0])
        
        result = 'SPAM' if prediction == 1 else 'LEGITIMATE'
        return result, confidence

def main():
    print('='*60)
    print('SPAM EMAIL CLASSIFIER - Machine Learning')
    print('='*60)
    
    classifier = SpamEmailClassifier()
    print('[0] Loading sample data...')
    emails, labels = classifier.load_sample_data()
    print(f'  Total emails: {len(emails)}')
    
    classifier.prepare_data(emails, labels)
    classifier.train_naive_bayes()
    classifier.train_svm()
    
    print('[4] Model Evaluation:')
    nb_results = classifier.evaluate_model(classifier.naive_bayes_model, 'Naive Bayes')
    print()
    svm_results = classifier.evaluate_model(classifier.svm_model, 'SVM')
    
    print('[5] Test Classifications:')
    test_emails = [
        'Your account has been compromised. Click here now.',
        'Meeting rescheduled to next Monday at 2 PM'
    ]
    
    for email in test_emails:
        nb_result, nb_conf = classifier.classify_email(email, 'naive_bayes')
        svm_result, svm_conf = classifier.classify_email(email, 'svm')
        print(f'Email: {email[:40]}...')
        print(f'  NB: {nb_result} ({nb_conf:.2f})')
        print(f'  SVM: {svm_result} ({svm_conf:.2f})')

if __name__ == '__main__':
    main()
