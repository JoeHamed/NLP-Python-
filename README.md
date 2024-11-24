# Natural Language Processing (NLP) - Restaurant Reviews Sentiment Analysis

This project demonstrates the use of Natural Language Processing (NLP) techniques to classify restaurant reviews as positive or negative. The workflow involves cleaning text data, creating a Bag-of-Words model, and training a Naive Bayes classifier for sentiment analysis.

---

## Dataset

The dataset, `Restaurant_Reviews.tsv`, contains 1,000 reviews from customers with the following structure:

| Review                            | Liked |
|-----------------------------------|-------|
| Wow... Loved this place.          | 1     |
| Crust is not good.                | 0     |
| Not tasty and the texture was...  | 0     |

- **Review**: Text of the customer's review.
- **Liked**: Binary target variable (1 = Positive, 0 = Negative).

---

## Workflow

### 1. Text Cleaning
- Removed special characters, numbers, and punctuation using Regular Expressions.
- Converted all text to lowercase.
- Removed stopwords (common words that add little semantic value) but retained the word *not* for sentiment purposes.
- Applied stemming to retain only the root form of words (e.g., "loved" → "love").

### 2. Bag-of-Words Model
- Used `CountVectorizer` to convert cleaned text into a numerical representation.
- Selected the 1,400 most frequent words for efficiency.

### 3. Splitting Data
- Split data into Training (80%) and Testing (20%) sets using `train_test_split`.

### 4. Training and Prediction
- Trained a **Naive Bayes classifier** on the training set.
- Predicted the test set results with a **73% accuracy**.

### 5. Confusion Matrix
Evaluated model performance:
#### [[55 42]
#### [12 91]]

- True Positives: 91
- True Negatives: 55
- False Positives: 42
- False Negatives: 12

---

## Predicting a Single Review
You can predict whether a single review is positive or negative using the trained model. Example:
```python
review = "I love this restaurant so much"
# Preprocess the review...
new_y_pred = classifier.predict(new_X_test)
if new_y_pred == 1:
    print('Positive')
else:
    print('Negative')
```
## How to Run
### Prerequisites
Ensure the following libraries are installed:
- numpy
- pandas
- matplotlib
- nltk
- scikit-learn
