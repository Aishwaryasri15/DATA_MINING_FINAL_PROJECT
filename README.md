## Thyroid Disease Prediction

This project focuses on the prediction and analysis of thyroid diseases, specifically hyperthyroidism and hypothyroidism, using machine learning models. The thyroid gland plays a crucial role in regulating metabolism through hormone release. Accurate diagnosis and prediction of thyroid diseases are essential for effective treatment and patient care.

### Dataset
The dataset used for this study is obtained from the UCI Machine Learning Repository, specifically focusing on hypothyroid data. The data has been preprocessed to ensure quality and relevance for analysis and model training.

### Project Architecture
The following steps outline the architecture and workflow of the project:

1. **Data Collection**
   - Obtain the dataset from the UCI Machine Learning Repository.
   
2. **Data Preprocessing**
   - Handle missing values using techniques like KNN Imputer.
   - Encode categorical variables using Label Encoding.
   - Scale numerical features using StandardScaler.
   
3. **Data Resampling**
   - Apply oversampling techniques like SMOTENC and RandomOverSampler to handle class imbalance.
   
4. **Feature Selection**
   - Use variance inflation factor (VIF) to identify and remove multicollinear features.
   
5. **Model Training**
   - Split the data into training and testing sets using `train_test_split`.
   - Train various machine learning models including SVM, K-NN, and Logistic Regression.
   
6. **Hyperparameter Tuning**
   - Optimize model hyperparameters using `GridSearchCV` and `RandomizedSearchCV`.
   
7. **Model Evaluation**
   - Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, ROC-AUC score.
   - Plot and analyze confusion matrices using `ConfusionMatrixDisplay`.
   
8. **Model Prediction**
   - Use the trained and optimized models to predict the probability of thyroid disease in new patient data.
   
9. **Report Generation**
   - Generate a comprehensive report of the data analysis and model performance using `pandas_profiling`.

### Machine Learning Models
Several machine learning algorithms were employed to predict the likelihood of a patient developing thyroid disease:
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (K-NN)**
- **Logistic Regression**

### Model Optimization
Hyperparameter tuning was performed using `GridSearchCV` and `RandomizedSearchCV` to optimize the K-NN classifier. The optimized models achieved an accuracy of approximately 95%.

### Objective
The objective of this project is to leverage machine learning techniques to improve the diagnosis and prediction of thyroid diseases, thereby aiding in better decision-making and treatment planning.

### Key Features
- Data preprocessing and purification to ensure high-quality inputs.
- Implementation of multiple classification models.
- Hyperparameter tuning for model optimization.
- High accuracy in predicting thyroid disease.

### Usage
The project includes scripts for data preprocessing, model training, evaluation, and hyperparameter tuning. Users can run these scripts to train models on the provided dataset and predict thyroid disease in patients.

### Conclusion
The K-Nearest Neighbors (KNN) classifier outperformed Logistic Regression and Support Vector Machines (SVM) in predicting thyroid disease. Hyperparameter tuning using GridSearchCV and RandomizedSearchCV further improved KNN's multiclass ROC AUC score from 0.9354 to 0.9431. These enhancements underscore the significance of model optimization in medical diagnosis, enhancing accuracy and reliability for patient care. KNN, after tuning, emerged as the most effective model for thyroid disease prediction.
