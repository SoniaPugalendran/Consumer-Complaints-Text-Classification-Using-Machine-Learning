`     `**Consumer Complaints Text Classification Using Machine Learning**


**Table of Content:**

- Introduction
- Dataset Used
- Technologies & Tools Used
- Model Architecture
- How to Run
- Results
- Methodology
- Future Enhancements
- Author

**Introduction:**
**\
`       `This project aims to build a multi-class text classification model for consumer complaint narratives.It categorizes complaints into four classes: Credit Reporting, Debt Collection, Consumer Loan, and Mortgage.The workflow includes data cleaning, EDA, text preprocessing, and feature extraction using TF-IDF.Multiple machine learning models like Logistic Regression, Linear SVM, XGBoost and Random Forest are trained.Models are evaluated using accuracy, precision, recall, F1-score, and confusion matrix.The best-performing model is used to predict the category of new complaint texts.

**Abstract:**
**\
`      `This project focuses on classifying consumer complaints from the U.S. Consumer Complaint Database into four categories: CreditReporting/Repair/Other, Debt Collection, Consumer Loan, and Mortgage. The workflow includes exploratory data analysis, text preprocessing (like lowercasing, stopword removal, and lemmatization), and feature extraction using TF-IDF vectorization. Multiple classification models including Logistic Regression, Linear SVM, and Random Forest were trained and evaluated. Among them, XGBoost performed best, achieving high accuracy and balanced F1-scores across classes. The solution emphasizes interpretability and effectiveness using traditional NLP techniques and is fully documented with screenshots and reproducible code in a Colab notebook.

` `Keyword: Text Classification, Multi-Class Classification, Machine Learning

**Dataset Used:**

- Name: Consumer Complaint** 
- Features: 18 features
- Categories Covered: Credit reporting, repair, or other,  Debt collection, Consumer Loan, Mortgage.
- Dataset Link: https://drive.google.com/file/d/1ePbxKMLoBsVJSIlAq8PAoNW3-odL3pdN/view?usp=sharing

**Technologies & Tools Used:**

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- SMOTE (for class balancing)
- Google Colab (for training)

**Model Architecture:**

- TF-IDF Vectorization for converting textual complaint narratives into numerical feature vectors.
- Logistic Regression for simple, interpretable linear classification.
- Linear SVM for maximizing class separation in high-dimensional text space.
- Random Forest for capturing non-linear relationships and feature importance.
- XGBoost for efficient, high-performance gradient boosting on sparse text features.
- Final model selected based on evaluation metrics like accuracy, precision, recall, and F1-score.

How to Run:

- **Clone the repository**

  https://github.com/[SoniaPugalendran]/Consumer-Complaint-Text-Classification.git

`          `cd Consumer-Complaint-Text-Classification

- Install the required libraries pip install -r requirements.txt
- Run the main Python file python main.py
- View the result

  `      `The script will load and preprocess the complaint data, train the models (Logistic Regression, SVM, Random Forest, XGBoost), and display accuracy, precision, recall, F1-score, and the confusion matrix for comparison.

**Result:**

`  `###Dataset Details(Size,Features)

`        `![Dataset Details](screenshots/1\_dataset\_details.png)



    ###Data Preprocessing Output

`        `![Preprocessing](screenshots/2\_data\_preprocessing.png)

` `###Class Distribution Output after Preprocessing(Split of Train and Test)

`        `![Class Distribution](screenshots/3\_class\_distribution.png)

` `###Class Distribution Output Graph

`        `![Class Distribution  Graph](screenshots/4\_class\_distribution\_graph.png)

` `###Handle Class Imbalance using SMOTE Technique Output

`        `![Class Balance](screenshots/5\_class\_balance\_using\_smote.png)

` `###Model Training Using Logistic Regression Classification Report

`        `![Logistic Regression](screenshots/6\_logistic\_regression.png)

` `###Model Training Using Logistic Regression Confusion Matrix

`        `![LogisticRegression](screenshots/7\_logistic\_regression\_confusion\_matrix.png)

###Model Training Using Random Forest Classification Report

`        `![Random Forest](screenshots/8\_random\_forest.png)

###Model Training Using Random Forest Confusion Matrix

`        `![Random Forest](screenshots/9\_random\_forest\_confusion\_matrix.png)

###Model Training Using Linear SVM Classification Report

`        `![Linear SVM](screenshots/10\_linear\_svm.png)



###Model Training Using Linear SVM Confusion Matrix



`        `![Linear SVM](screenshots/11\_linear\_svm\_confusion\_matrix.png)

###Model Training Using XGBoost Classification Report

`        `![XGBoost](screenshots/12\_xgboost.png)

###Model Training Using XGBoost Confusion Matrix

`        `![XGBoost](screenshots/13\_xgboost\_confusion\_matrix.png)

###Model Comparison Output

`        `![Model Comparison](screenshots/14\_model\_comparison.png)

###Complaints Prediction Output

`        `![Complaints Prediction](screenshots/15\_complaint\_prediction.png)


## <a name="_evr7g8m0htmn"></a>**Methodology:**
- Cleaned complaint text using lowercasing, punctuation removal, stopword removal, and lemmatization
- Converted text into numerical features using TF-IDF vectorization
- Encoded target complaint categories into numerical labels (0–3)
- Applied Stratified Train-Test Split to maintain class balance
- Trained four models: Logistic Regression, Linear SVM, Random Forest, and XGBoost
- Evaluated models using Accuracy, Precision, Recall, F1-score, and Confusion Matrix for comparison

**Saved Models:**

`   `**https://drive.google.com/drive/folders/1oHQ\_TldZHbrgX93L86vz-srghwMWNvKs?usp=sharing**

### <a name="_hveffotnc24p"></a>**Future Enhancements:**
- Integration of a Flask API for real-time complaint category prediction from user input
- Hyperparameter tuning using  Bayesian optimization to improve model performance
- Incorporation of deep learning models (e.g., LSTM, CNN) to capture contextual patterns in complaint narratives
- Model retraining pipeline using updated complaint data for continuous improvement

Author

SoniaPugalendran

[SoniaPugalendran/](https://github.com/SoniaPugalendran/Network-Intrusion-Detection-System-Using-DeepLearning)Consumer-Complaints-Text-Classification-Using-Machine-Learning

Date:11/06/2025

