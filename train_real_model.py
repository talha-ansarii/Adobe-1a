#!/usr/bin/env python3
"""
PDF Heading Detection Model Trainer with Real Dataset
Trains models using real labeled data for production use
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import ssl
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Handle SSL certificate issues for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk import pos_tag, word_tokenize
from collections import defaultdict
import re


class ProductionHeadingDetector:
    """Production-ready heading detector trained on real data"""
    
    def __init__(self, model_save_path=None):
        self.model_save_path = model_save_path or "heading_detector_model.pkl"
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.dt_classifier = None
        self.ensemble_classifier = None
        self.selected_features = None
        self.feature_columns = None
        
    def load_and_prepare_dataset(self, dataset_file):
        """Load and prepare the labeled dataset"""
        print(f"Loading dataset from: {dataset_file}")
        
        # Load dataset
        if dataset_file.endswith('.csv'):
            df = pd.read_csv(dataset_file)
        else:
            raise ValueError("Dataset must be a CSV file")
        
        # Check required columns
        required_cols = ['text', 'is_heading', 'font_size', 'is_bold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"Dataset loaded: {len(df)} samples")
        print(f"Headings: {df['is_heading'].sum()} ({df['is_heading'].mean():.1%})")
        
        return df
    
    def engineer_features(self, df):
        """Engineer comprehensive features from the dataset"""
        print("Engineering features...")
        
        # Copy dataframe to avoid modifying original
        df = df.copy()
        
        # Basic text features
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        df['char_count'] = df['text'].apply(lambda x: len(str(x)))
        df['avg_word_length'] = df['char_count'] / df['word_count'].replace(0, 1)
        
        # Font features
        df['relative_font_size'] = df['font_size'] / df.get('font_threshold', df['font_size'].median())
        df['font_threshold_flag'] = (df['font_size'] > df.get('font_threshold', df['font_size'].median())).astype(int)
        
        # Text case features
        df['is_all_caps'] = df['text'].apply(lambda x: str(x).isupper()).astype(int)
        df['is_title_case'] = df['text'].apply(lambda x: str(x).istitle()).astype(int)
        df['is_lower_case'] = df['text'].apply(lambda x: str(x).islower()).astype(int)
        
        # Structural features
        df['starts_with_number'] = df['text'].apply(lambda x: bool(re.match(r'^\s*\d+', str(x)))).astype(int)
        df['ends_with_colon'] = df['text'].apply(lambda x: str(x).strip().endswith(':')).astype(int)
        df['has_punctuation'] = df['text'].apply(lambda x: any(c in str(x) for c in '.,;!?')).astype(int)
        
        # Length-based features
        df['is_short'] = (df['word_count'] <= 10).astype(int)
        df['is_very_short'] = (df['word_count'] <= 5).astype(int)
        df['is_long'] = (df['word_count'] > 20).astype(int)
        
        # POS tagging features
        print("Extracting POS features...")
        pos_features = df['text'].apply(self.extract_pos_features)
        pos_df = pd.DataFrame(pos_features.tolist())
        df = pd.concat([df, pos_df], axis=1)
        
        # Position features (if available)
        if 'y0' in df.columns and 'page_number' in df.columns:
            # Normalize position within each page
            for page in df['page_number'].unique():
                page_mask = df['page_number'] == page
                page_data = df[page_mask]
                
                if len(page_data) > 1:
                    df.loc[page_mask, 'relative_y_position'] = (
                        (page_data['y0'].max() - page_data['y0']) / 
                        (page_data['y0'].max() - page_data['y0'].min())
                    )
                else:
                    df.loc[page_mask, 'relative_y_position'] = 0.5
        
        # Fill missing values
        df = df.fillna(0)
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    def extract_pos_features(self, text):
        """Extract Part-of-Speech features from text"""
        try:
            tokens = word_tokenize(str(text).lower())
            pos_tags = pos_tag(tokens)
            
            pos_counts = defaultdict(int)
            total_words = len(tokens)
            
            for word, pos in pos_tags:
                if pos.startswith('VB'):  # Verbs
                    pos_counts['verb_ratio'] += 1
                elif pos.startswith('NN'):  # Nouns
                    pos_counts['noun_ratio'] += 1
                elif pos.startswith('JJ'):  # Adjectives
                    pos_counts['adjective_ratio'] += 1
                elif pos.startswith('RB'):  # Adverbs
                    pos_counts['adverb_ratio'] += 1
                elif pos in ['PRP', 'PRP$']:  # Pronouns
                    pos_counts['pronoun_ratio'] += 1
                elif pos == 'CD':  # Cardinal numbers
                    pos_counts['number_ratio'] += 1
            
            # Convert to ratios
            for key in pos_counts:
                pos_counts[key] = pos_counts[key] / total_words if total_words > 0 else 0
            
            return pos_counts
            
        except:
            return {'verb_ratio': 0, 'noun_ratio': 0, 'adjective_ratio': 0, 
                    'adverb_ratio': 0, 'pronoun_ratio': 0, 'number_ratio': 0}
    
    def prepare_training_data(self, df):
        """Prepare data for machine learning"""
        # Identify feature columns (exclude text and target)
        exclude_cols = ['text', 'is_heading', 'pdf_file', 'heading_level', 'manual_label', 
                       'auto_confidence', 'font_name', 'is_synthetic']
        
        feature_columns = [col for col in df.columns 
                          if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_columns].copy()
        y = df['is_heading'].copy()
        
        # Handle missing values and infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Training features: {len(feature_columns)}")
        print(f"Training samples: {len(X)}")
        print(f"Positive samples: {y.sum()} ({y.mean():.1%})")
        
        self.feature_columns = feature_columns
        return X, y
    
    def train_models(self, X, y, test_size=0.2, use_smote=True):
        """Train multiple models and select the best"""
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        if use_smote and len(np.unique(y_train)) > 1:
            print("Applying SMOTE for class balance...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {len(X_train_balanced)} samples")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to DataFrame to maintain column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        # Feature selection
        print("Selecting optimal features...")
        selector = RFECV(
            estimator=DecisionTreeClassifier(random_state=42),
            step=1,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1
        )
        
        X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
        X_test_selected = selector.transform(X_test_scaled)
        
        self.feature_selector = selector
        self.selected_features = [self.feature_columns[i] for i in range(len(self.feature_columns)) if selector.support_[i]]
        
        print(f"Selected {len(self.selected_features)} features: {self.selected_features}")
        
        # Train Decision Tree with hyperparameter tuning
        print("\nTraining Decision Tree...")
        dt_param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_features': ['sqrt', 'log2', None]
        }
        
        dt_grid = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            dt_param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        dt_grid.fit(X_train_selected, y_train_balanced)
        self.dt_classifier = dt_grid.best_estimator_
        
        print(f"Best DT parameters: {dt_grid.best_params_}")
        print(f"Best DT CV score: {dt_grid.best_score_:.4f}")
        
        # Train Ensemble
        print("\nTraining Ensemble Classifier...")
        base_classifiers = [
            ('dt', DecisionTreeClassifier(**dt_grid.best_params_, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('svm', SVC(probability=True, random_state=42)),
            ('nb', GaussianNB())
        ]
        
        self.ensemble_classifier = VotingClassifier(
            estimators=base_classifiers,
            voting='soft'
        )
        
        self.ensemble_classifier.fit(X_train_selected, y_train_balanced)
        
        # Evaluate models
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Decision Tree evaluation
        dt_pred = self.dt_classifier.predict(X_test_selected)
        dt_scores = {
            'accuracy': accuracy_score(y_test, dt_pred),
            'precision': precision_score(y_test, dt_pred),
            'recall': recall_score(y_test, dt_pred),
            'f1': f1_score(y_test, dt_pred)
        }
        
        print("Decision Tree Performance:")
        for metric, score in dt_scores.items():
            print(f"  {metric.capitalize()}: {score:.4f}")
        
        # Ensemble evaluation
        ensemble_pred = self.ensemble_classifier.predict(X_test_selected)
        ensemble_scores = {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred),
            'recall': recall_score(y_test, ensemble_pred),
            'f1': f1_score(y_test, ensemble_pred)
        }
        
        print("\nEnsemble Performance:")
        for metric, score in ensemble_scores.items():
            print(f"  {metric.capitalize()}: {score:.4f}")
        
        # Feature importance
        if hasattr(self.dt_classifier, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.dt_classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
        
        # Confusion Matrix
        print(f"\nConfusion Matrix (Decision Tree):")
        cm = confusion_matrix(y_test, dt_pred)
        print(cm)
        
        return {
            'dt_scores': dt_scores,
            'ensemble_scores': ensemble_scores,
            'feature_importance': feature_importance if 'feature_importance' in locals() else None,
            'test_predictions': dt_pred,
            'test_true': y_test
        }
    
    def save_model(self):
        """Save the trained model to disk"""
        model_data = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'dt_classifier': self.dt_classifier,
            'ensemble_classifier': self.ensemble_classifier,
            'selected_features': self.selected_features,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, self.model_save_path)
        print(f"\nModel saved to: {self.model_save_path}")
    
    def load_model(self):
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(self.model_save_path)
            
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.dt_classifier = model_data['dt_classifier']
            self.ensemble_classifier = model_data['ensemble_classifier']
            self.selected_features = model_data['selected_features']
            self.feature_columns = model_data['feature_columns']
            
            print(f"Model loaded from: {self.model_save_path}")
            return True
        except FileNotFoundError:
            print(f"Model file not found: {self.model_save_path}")
            return False
    
    def predict_pdf_headings(self, pdf_path, use_ensemble=True):
        """Predict headings in a new PDF using the trained model"""
        from dataset_creator import DatasetCreator
        
        # Extract features from PDF
        creator = DatasetCreator()
        elements = creator.extract_text_elements(pdf_path)
        
        if not elements:
            return []
        
        # Convert to DataFrame and engineer features
        df = pd.DataFrame(elements)
        df_features = self.engineer_features(df)
        
        # Prepare features for prediction
        X = df_features[self.feature_columns].fillna(0)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Make predictions
        if use_ensemble and self.ensemble_classifier:
            predictions = self.ensemble_classifier.predict(X_selected)
            probabilities = self.ensemble_classifier.predict_proba(X_selected)[:, 1]
        else:
            predictions = self.dt_classifier.predict(X_selected)
            probabilities = self.dt_classifier.predict_proba(X_selected)[:, 1]
        
        # Add predictions to DataFrame
        df_features['predicted_heading'] = predictions
        df_features['heading_probability'] = probabilities
        
        return df_features


def main():
    """Main training function"""
    print("PDF Heading Detection Model Trainer")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_file = "/Users/talhaansari/Developer/Adobe/1a new/training_dataset.csv"
    
    if not Path(dataset_file).exists():
        print(f"❌ Dataset not found: {dataset_file}")
        print("Please run dataset_creator.py first to create the training dataset.")
        return
    
    # Initialize trainer
    model_path = "/Users/talhaansari/Developer/Adobe/1a new/production_heading_model.pkl"
    trainer = ProductionHeadingDetector(model_path)
    
    try:
        # Load and prepare dataset
        df = trainer.load_and_prepare_dataset(dataset_file)
        
        # Engineer features
        df_features = trainer.engineer_features(df)
        
        # Prepare training data
        X, y = trainer.prepare_training_data(df_features)
        
        # Train models
        results = trainer.train_models(X, y)
        
        # Save model
        trainer.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Model saved to: {model_path}")
        print(f"📊 Best F1 Score: {max(results['dt_scores']['f1'], results['ensemble_scores']['f1']):.4f}")
        print("🚀 Ready for production use!")
        
        # Test on sample PDF
        print("\nTesting on sample PDF...")
        sample_pdf = "/Users/talhaansari/Developer/Adobe/1a new/input/example.pdf"
        if Path(sample_pdf).exists():
            result_df = trainer.predict_pdf_headings(sample_pdf)
            headings = result_df[result_df['predicted_heading'] == 1]
            print(f"Detected {len(headings)} headings in {Path(sample_pdf).name}")
            
            for i, (_, heading) in enumerate(headings.head(5).iterrows(), 1):
                print(f"{i}. {heading['text'][:60]}... (confidence: {heading['heading_probability']:.1%})")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
