# Titanic Exploratory Data Analysis

## Project Overview

This project performs Exploratory Data Analysis (EDA) on the Titanic passenger dataset.

The main objective is to understand the dataset, identify patterns, handle data issues, and analyze factors that influenced passenger survival.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Analysis Performed

The following analysis was performed:

- Dataset inspection
- Statistical summary
- Missing value analysis
- Duplicate checking
- Data cleaning
- Survival count analysis
- Survival percentage analysis
- Survival based on gender
- Survival based on passenger class
- Data visualization

## Key Findings

### Overall Survival

- 455 passengers did not survive.
- 320 passengers survived.
- 58.71% of passengers did not survive.
- 41.29% of passengers survived.

### Survival by Gender

- Female survival rate: 73.97%
- Male survival rate: 21.53%

This shows that female passengers had a significantly higher survival rate.

### Survival by Passenger Class

- First Class: 63.33%
- Second Class: 50.61%
- Third Class: 25.94%

Passengers in higher classes had higher survival rates compared to passengers in Third Class.

## Project Structure

```text
EDA_Project/
│
├── results/
│   └── visualization files
│
├── eda_analysis.py
├── visualizations.py
├── requirements.txt
├── README.md
└── .gitignore
