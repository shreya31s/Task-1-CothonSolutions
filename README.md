# Healthcare Data Analysis and Disease Prediction

## Project Description
This project focuses on healthcare data analysis and disease prediction, specifically aiming to predict the likelihood of diabetes in individuals based on various health metrics. Leveraging machine learning techniques, this application provides an interactive interface for users to input their health data and receive predictions regarding their risk of developing diabetes. 

### Objectives
- Analyze the Kaggle "Diabetes Dataset" to understand key features contributing to diabetes risk.
- Build and evaluate multiple machine learning models for accurate predictions.
- Develop a user-friendly web application to facilitate easy data input and visualization of results.
- Provide insights through data visualizations, model performance metrics, and feature importance analysis.

## Dataset
The project utilizes the **Diabetes Dataset** provided by Akshay Dattatray Khare on Kaggle. This dataset includes various health-related attributes such as:
- Glucose levels
- Blood pressure
- Body mass index (BMI)
- Age
- Insulin levels
- Skin thickness

The dataset can be accessed [here](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset).

## Technologies Used
This project employs a variety of technologies to facilitate both the backend and frontend development:
- **Python**: Core programming language for data processing and machine learning.
- **FastAPI**: Framework for building the backend RESTful API.
- **Uvicorn**: ASGI server for serving the FastAPI application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and data handling.
- **Scikit-learn**: For implementing machine learning algorithms.
- **XGBoost**: For advanced gradient boosting algorithm.
- **React.js**: Frontend framework for building the user interface.
- **Tailwind CSS**: Utility-first CSS framework for styling the application.

## Features
- **User Input Form**: Users can input their health metrics through a simple and intuitive form.
- **Prediction Output**: The application provides a prediction of diabetes risk based on the input data.
- **Data Visualization**: Users can visualize model performance metrics and feature importance using various graphs and charts.
- **Responsive Design**: The application is designed to be fully responsive, ensuring

## Model Training and Evaluation
The project implements several machine learning models to predict diabetes, including:
- Support Vector Machine (SVM)
- Decision Trees
- Random Forest
- XGBoost
The models are trained on the training dataset, and their performance is evaluated using metrics such as:

- ROC-AUC: To measure the area under the Receiver Operating Characteristic curve.
- Confusion Matrix: To visualize true positives, false positives, true negatives, and false negatives.
- Precision and Recall: To evaluate the accuracy of the positive class predictions.


## Results and Insights
The application not only provides predictions but also offers visual insights into the data. Users can view the importance of different features contributing to the prediction and understand how their inputs affect the results.

### Contributer
Shreya. S - www.linkedin.com/in/shreya-suresh-620922256
