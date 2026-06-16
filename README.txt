# Emotion Classification Project

## Project Description

This project focuses on emotion classification in short text using Natural Language Processing (NLP) and Machine Learning techniques.

The main objective is to automatically classify text messages into emotional categories based on their content.

The project includes a complete machine learning pipeline:
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Multiple machine learning models
- A manually implemented neural network using NumPy
- Hyperparameter tuning using GridSearchCV
- Model evaluation using multiple classification metrics
- Comparative analysis of results

---

## Dataset

The dataset used in this project is a labeled emotion dataset containing short text samples associated with emotional categories.

The emotion classes are:
- Joy
- Sadness
- Love
- Anger
- Fear
- Surprise

### Preprocessing steps applied:
- Conversion to lowercase
- Removal of punctuation and special characters
- Removal of stopwords (NLTK + custom stopwords)
- Tokenization and text cleaning
- Removal of empty or null values
- Creation of a balanced subset of the dataset

The dataset is split as follows:
- 80% training set
- 20% testing set

Processed data is stored in:
data/processed/

---

## Project Structure

```text
data/
 ├── raw/
 ├── processed/

models/
src/
main.py
requirements.txt
README.md

Machine Learning Models

The following models were implemented and compared:

Logistic Regression
Support Vector Machine (LinearSVC)
Multinomial Naive Bayes
Neural Network (manual implementation using NumPy)

All models were trained using the same TF-IDF feature representation for fair comparison.

Feature Extraction

Text data was converted into numerical form using TF-IDF (Term Frequency - Inverse Document Frequency).

Configuration:

Maximum features: 5000
Sparse matrix representation
Used as input for all models
Experimental Setup
Dataset split: 80% training / 20% testing
Same preprocessing pipeline for all experiments
GridSearchCV used for SVM hyperparameter tuning
Cross-validation: 5-fold
Best SVM parameter: C = 0.1
Evaluation Metrics

The models were evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

The F1-score was used as the main metric for comparison.

Results

Best performing models:

Logistic Regression: ~0.912 accuracy
Support Vector Machine: ~0.911 accuracy

Naive Bayes:

~0.87 accuracy

Neural Network:

Lower performance due to simple architecture
How to Run the Project
Install dependencies
pip install -r requirements.txt
Run project
python main.py
Steps executed:
Load dataset
Preprocess text data
Extract TF-IDF features
Train models
Evaluate performance
Display results
Requirements

Main libraries used:

numpy
pandas
scikit-learn
nltk
matplotlib
seaborn

All dependencies are included in:

requirements.txt

Limitations
TF-IDF does not capture semantic meaning
Difficulty handling sarcasm or ambiguous text
Simple neural network architecture
Dataset is limited to a balanced subset
Future Improvements
Using BERT or Transformer models
Word embeddings (Word2Vec, GloVe)
Improved neural networks (LSTM, deep learning)
Larger dataset for better generalization
Web interface for real-time prediction
Conclusion

This project demonstrates that machine learning techniques can effectively classify emotions in short text messages.

Classical models such as Logistic Regression and SVM achieved strong performance when combined with TF-IDF.

The project successfully achieves its objective and provides a reproducible NLP pipeline.

Label Encoding

The dataset uses the following encoding:

0 = Sadness
1 = Joy
2 = Love
3 = Anger
4 = Fear
5 = Surprise
Author

Alexandra Rădulescu