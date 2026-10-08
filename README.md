# 🏠 Pakistan House Price Prediction

A Machine Learning project that predicts house prices in major Pakistani cities using Zameen.com data.

## Problem Statement
Housing prices in Pakistan are inconsistent. This project predicts property prices based on location, size, bedrooms, bathrooms, and property type.

## Dataset
- For_EDA_dataset.csv — Raw data (153,430 records)
- House_Price_dataset.csv — Original dataset (168,446 records)
- Cleaned_data_for_model.csv — Cleaned data for modeling (99,499 records)

Source: Zameen.com via Kaggle | Cities: Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad

## Models Used
- Linear Regression — R² ~0.55
- Gradient Boosting — R² ~0.78
- Random Forest (Best) — R² ~0.80

## How to Run

Install libraries:
pip install pandas numpy matplotlib seaborn scikit-learn jupyter streamlit

Run Notebook:
jupyter notebook
Open Pakistan_House_Price_Prediction.ipynb → Kernel → Restart & Run All

Run Web App:
streamlit run app.py

## Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit

## Author
Machine Learning Course Project
