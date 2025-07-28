#!/usr/bin/env python3
"""
PDF Heading Detection using Hybrid Deep Learning Approach
Based on the implementation guide for achieving 95.83% to 96.95% accuracy
"""

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import ssl
import os
from pathlib import Path

# Handle SSL certificate issues for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk import pos_tag, word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


class PDFFeatureExtractor:
    """Extract comprehensive features from PDF for heading detection"""
    
    def __init__(self):
        self.features = []
        
    def extract_pdf_features(self, pdf_path):
        """Extract comprehensive features from PDF for heading detection"""
        print(f"Processing {pdf_path}...")
        doc = fitz.open(pdf_path)
        features = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            # Calculate document-level font threshold
            font_sizes = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
            
            # Most frequent font size as threshold (assumption: body text)
            font_threshold = max(set(font_sizes), key=font_sizes.count) if font_sizes else 12
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        # Merge spans in same line
                        line_text = " ".join([span["text"].strip() for span in line["spans"]])
                        if not line_text or len(line_text.strip()) == 0:
                            continue
                        
                        # Calculate average properties for the line
                        avg_font_size = np.mean([span["size"] for span in line["spans"]])
                        is_bold = any(bool(span["flags"] & 16) for span in line["spans"])
                        
                        # Extract bounding box for position features
                        y0 = min(span["bbox"][1] for span in line["spans"])
                        x0 = min(span["bbox"][0] for span in line["spans"])
                        
                        feature_dict = self.extract_text_features(
                            line_text, avg_font_size, is_bold, font_threshold, 
                            y0, x0, page_num + 1
                        )
                        features.append(feature_dict)
        
        doc.close()
        return pd.DataFrame(features)

    def extract_text_features(self, text, font_size, is_bold, font_threshold, y_pos, x_pos, page_num):
        """Extract comprehensive features from text element"""
        features = {}
        
        # Basic text properties
        features['text'] = text
        features['characters'] = len(text)
        features['words'] = len(text.split())
        features['page_number'] = page_num
        
        # Font and visual features
        features['font_size'] = font_size
        features['is_bold'] = int(is_bold)
        features['font_threshold_flag'] = int(font_size > font_threshold)
        
        # Position and spacing features
        features['y_position'] = y_pos
        features['x_position'] = x_pos  # For indentation analysis
        features['relative_font_size'] = font_size / font_threshold if font_threshold > 0 else 1
        
        # Text case analysis
        if text.isupper():
            features['text_case'] = 1  # All uppercase
        elif text.istitle():
            features['text_case'] = 2  # Title case
        elif text.islower():
            features['text_case'] = 0  # All lowercase
        else:
            features['text_case'] = 3  # Mixed case
        
        # POS tagging features
        pos_features = self.extract_pos_features(text)
        features.update(pos_features)
        
        # Structural features
        features['starts_with_number'] = int(bool(re.match(r'^\d+\.?\s', text)))
        features['ends_with_colon'] = int(text.strip().endswith(':'))
        features['has_punctuation'] = int(any(c in text for c in '.,;!?'))
        
        return features

    def extract_pos_features(self, text):
        """Extract Part-of-Speech features from text"""
        try:
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            pos_counts = defaultdict(int)
            for word, pos in pos_tags:
                if pos.startswith('VB'):  # Verbs
                    pos_counts['verbs'] += 1
                elif pos.startswith('NN'):  # Nouns
                    pos_counts['nouns'] += 1
                elif pos.startswith('JJ'):  # Adjectives
                    pos_counts['adjectives'] += 1
                elif pos.startswith('RB'):  # Adverbs
                    pos_counts['adverbs'] += 1
                elif pos in ['PRP', 'PRP$']:  # Pronouns
                    pos_counts['pronouns'] += 1
                elif pos == 'CD':  # Cardinal numbers
                    pos_counts['cardinal_numbers'] += 1
                elif pos == 'CC':  # Coordinating conjunctions
                    pos_counts['coordinating_conjunctions'] += 1
            
            return pos_counts
        except:
            return {'verbs': 0, 'nouns': 0, 'adjectives': 0, 'adverbs': 0, 
                    'pronouns': 0, 'cardinal_numbers': 0, 'coordinating_conjunctions': 0}

    def create_advanced_features(self, df):
        """Create advanced features for improved heading detection"""
        
        # Normalize position features within each page
        for page in df['page_number'].unique():
            page_mask = df['page_number'] == page
            page_data = df[page_mask]
            
            if len(page_data) > 1:
                # Relative position on page (0-1 scale)
                df.loc[page_mask, 'relative_y_position'] = (
                    (page_data['y_position'].max() - page_data['y_position']) / 
                    (page_data['y_position'].max() - page_data['y_position'].min())
                )
            else:
                df.loc[page_mask, 'relative_y_position'] = 0.5
        
        # Text pattern features
        df['avg_word_length'] = df['characters'] / df['words'].replace(0, 1)
        df['word_density'] = df['words'] / df['characters'].replace(0, 1)
        
        # Ratio features (important for normalization across documents)
        df['verb_ratio'] = df['verbs'] / df['words'].replace(0, 1)
        df['noun_ratio'] = df['nouns'] / df['words'].replace(0, 1)
        df['adjective_ratio'] = df['adjectives'] / df['words'].replace(0, 1)
        
        # Length-based features (headings tend to be shorter)
        df['is_short'] = (df['words'] <= 10).astype(int)
        df['is_very_short'] = (df['words'] <= 5).astype(int)
        
        return df


class HeadingClassifier:
    """Main class for heading detection using hybrid deep learning approach"""
    
    def __init__(self):
        self.feature_extractor = PDFFeatureExtractor()
        self.dt_classifier = None
        self.ensemble_classifier = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.selected_features = None
        
    def apply_recursive_feature_elimination(self, X, y, estimator=None, cv_folds=5):
        """Apply RFE with cross-validation to select optimal features"""
        
        if estimator is None:
            estimator = DecisionTreeClassifier(
                criterion='gini',
                random_state=42,
                min_samples_split=2,
                min_samples_leaf=3
            )
        
        # Recursive Feature Elimination with Cross-Validation
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1
        )
        
        # Fit the selector
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.support_].tolist()
        X_selected = selector.transform(X)
        
        print(f"Optimal number of features: {selector.n_features_}")
        print(f"Selected features: {selected_features}")
        
        return X_selected, selected_features, selector

    def prepare_features_for_training(self, df, target_column='is_heading'):
        """Prepare features for machine learning"""
        
        # Remove text column and other non-numeric features for ML
        feature_columns = [col for col in df.columns 
                          if col not in ['text', target_column, 'page_number'] 
                          and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_columns].fillna(0)
        y = df[target_column] if target_column in df.columns else np.zeros(len(df))
        
        # Handle missing values and infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X, y, feature_columns

    def train_decision_tree(self, X_train, y_train, feature_selection=True):
        """Train the decision tree with optional feature selection"""
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        # Feature selection
        if feature_selection:
            X_train_selected, self.selected_features, self.feature_selector = \
                self.apply_recursive_feature_elimination(X_train_scaled, y_train)
        else:
            X_train_selected = X_train_scaled
            self.selected_features = X_train.columns.tolist()
        
        # Hyperparameter tuning
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Grid search with cross-validation
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            dt, param_grid, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_selected, y_train)
        
        # Best classifier
        self.dt_classifier = grid_search.best_estimator_
        
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        
        return self

    def create_base_classifiers(self):
        """Create base classifiers for ensemble"""
        classifiers = {
            'decision_tree': DecisionTreeClassifier(
                criterion='gini',
                max_depth=7,
                min_samples_split=2,
                min_samples_leaf=3,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'naive_bayes': GaussianNB()
        }
        return classifiers

    def train_ensemble(self, X_train, y_train, voting='soft', feature_selection=True):
        """Train ensemble classifier"""
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        # Feature selection
        if feature_selection:
            X_train_selected, selected_features, self.feature_selector = \
                self.apply_recursive_feature_elimination(X_train_scaled, y_train)
        else:
            X_train_selected = X_train_scaled
        
        # Create base classifiers
        base_classifiers = self.create_base_classifiers()
        
        # Create ensemble
        estimators = [(name, clf) for name, clf in base_classifiers.items()]
        
        self.ensemble_classifier = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
        
        # Train ensemble
        self.ensemble_classifier.fit(X_train_selected, y_train)
        
        return self

    def predict(self, X_test, use_ensemble=False):
        """Make predictions on test data"""
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        if self.feature_selector:
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        else:
            X_test_selected = X_test_scaled
        
        if use_ensemble and self.ensemble_classifier:
            return self.ensemble_classifier.predict(X_test_selected)
        else:
            return self.dt_classifier.predict(X_test_selected)

    def predict_proba(self, X_test, use_ensemble=False):
        """Get prediction probabilities"""
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        if self.feature_selector:
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        else:
            X_test_selected = X_test_scaled
        
        if use_ensemble and self.ensemble_classifier:
            return self.ensemble_classifier.predict_proba(X_test_selected)
        else:
            return self.dt_classifier.predict_proba(X_test_selected)

    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.dt_classifier and self.selected_features:
            importance_df = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.dt_classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None

    def simulate_heading_labels(self, df):
        """
        Simulate heading labels based on heuristic rules
        This is a simplified approach for demonstration
        """
        labels = []
        
        for _, row in df.iterrows():
            is_heading = 0
            
            # Heuristic rules for identifying potential headings
            if (row['font_size'] > row['relative_font_size'] * 12 and  # Larger font
                row['is_bold'] == 1 and  # Bold text
                row['words'] <= 10 and  # Short text
                row['characters'] <= 100):  # Not too long
                is_heading = 1
            
            # Additional rules
            elif (row['text_case'] == 1 and  # All uppercase
                  row['words'] <= 8 and
                  row['is_bold'] == 1):
                is_heading = 1
            
            # Numbered sections
            elif (row['starts_with_number'] == 1 and
                  row['words'] <= 12 and
                  row['font_size'] >= row['relative_font_size'] * 10):
                is_heading = 1
            
            labels.append(is_heading)
        
        return labels

    def analyze_pdf(self, pdf_path):
        """Analyze a single PDF and detect headings"""
        # Extract features
        df = self.feature_extractor.extract_pdf_features(pdf_path)
        
        if df.empty:
            print(f"No text found in {pdf_path}")
            return None
        
        # Create advanced features
        df = self.feature_extractor.create_advanced_features(df)
        
        # Simulate labels for training (in real scenario, you'd have labeled data)
        df['is_heading'] = self.simulate_heading_labels(df)
        
        print(f"Found {len(df)} text elements in {pdf_path}")
        print(f"Detected {df['is_heading'].sum()} potential headings")
        
        # Prepare features
        X, y, feature_columns = self.prepare_features_for_training(df)
        
        if X.empty or len(np.unique(y)) < 2:
            print("Insufficient data for training")
            df['predicted_heading'] = 0
            df['heading_probability'] = 0.0
            return df
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        try:
            X_balanced, y_balanced = smote.fit_resample(X, y)
        except ValueError:
            print("SMOTE failed, using original data")
            X_balanced, y_balanced = X, y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_balanced
        )
        
        # Train models
        print("\n=== Training Decision Tree ===")
        self.train_decision_tree(X_train, y_train)
        
        print("\n=== Training Ensemble Classifier ===")
        self.train_ensemble(X_train, y_train, voting='soft')
        
        # Evaluate models
        print("\n=== Evaluation Results ===")
        
        # Decision Tree predictions
        dt_predictions = self.predict(X_test, use_ensemble=False)
        dt_accuracy = accuracy_score(y_test, dt_predictions)
        dt_precision = precision_score(y_test, dt_predictions)
        dt_recall = recall_score(y_test, dt_predictions)
        dt_f1 = f1_score(y_test, dt_predictions)
        
        print(f"Decision Tree - Accuracy: {dt_accuracy:.4f}, Precision: {dt_precision:.4f}, "
              f"Recall: {dt_recall:.4f}, F1: {dt_f1:.4f}")
        
        # Ensemble predictions
        ensemble_predictions = self.predict(X_test, use_ensemble=True)
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        ensemble_precision = precision_score(y_test, ensemble_predictions)
        ensemble_recall = recall_score(y_test, ensemble_predictions)
        ensemble_f1 = f1_score(y_test, ensemble_predictions)
        
        print(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, Precision: {ensemble_precision:.4f}, "
              f"Recall: {ensemble_recall:.4f}, F1: {ensemble_f1:.4f}")
        
        # Feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            print("\n=== Top 10 Most Important Features ===")
            print(importance_df.head(10))
        
        # Apply predictions to original data
        if not X.empty:
            original_predictions = self.predict(X, use_ensemble=True)
            df['predicted_heading'] = original_predictions
            df['heading_probability'] = self.predict_proba(X, use_ensemble=True)[:, 1]
        else:
            df['predicted_heading'] = 0
            df['heading_probability'] = 0.0
        
        return df

    def process_pdf_folder(self, folder_path):
        """Process all PDFs in a folder"""
        folder_path = Path(folder_path)
        pdf_files = list(folder_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return
        
        results = {}
        
        for pdf_file in pdf_files:
            print(f"\n{'='*60}")
            print(f"Processing: {pdf_file.name}")
            print('='*60)
            
            try:
                result_df = self.analyze_pdf(str(pdf_file))
                if result_df is not None:
                    results[pdf_file.name] = result_df
                    
                    # Save results
                    output_file = folder_path / f"{pdf_file.stem}_heading_analysis.csv"
                    result_df.to_csv(output_file, index=False)
                    print(f"Results saved to: {output_file}")
                    
                    # Display detected headings
                    headings = result_df[result_df['predicted_heading'] == 1]['text'].tolist()
                    print(f"\nDetected Headings ({len(headings)}):")
                    for i, heading in enumerate(headings, 1):
                        print(f"{i:2d}. {heading}")
                        
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {str(e)}")
        
        return results


def main():
    """Main function to run the PDF heading detection"""
    # Initialize the classifier
    classifier = HeadingClassifier()
    
    # Process all PDFs in the input folder
    input_folder = "/Users/talhaansari/Developer/Adobe/1a new/input"
    results = classifier.process_pdf_folder(input_folder)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    
    for pdf_name, df in results.items():
        if 'predicted_heading' in df.columns:
            predicted_headings = df[df['predicted_heading'] == 1]
            print(f"{pdf_name}: {len(predicted_headings)} headings detected out of {len(df)} text elements")
        else:
            print(f"{pdf_name}: No headings detected (insufficient training data)")


if __name__ == "__main__":
    main()
