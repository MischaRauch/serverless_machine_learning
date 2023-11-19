# Serverless Machine Learning
This repository contains resources and code for implementing serverless machine learning solutions, with a special focus on integrating models from Hugging Face, a popular hub for machine learning models. 

## Overview
Serverless architectures enable the deployment of applications and services without managing the underlying infrastructure. This project focuses on deploying machine learning models as serverless functions, which can scale automatically with demand and minimize operational costs. Special emphasis is given to Hugging Face models due to their wide usage and versatility in the machine learning community.

## Getting Started
### Prerequisites
- Knowledge of machine learning concepts.
- Familiarity with serverless architecture.
- Experience with Python, as many Hugging Face models are Python-based.
- An account on Hugging Face (optional, for accessing certain models).
## Installation
1. Clone the repository:
```git clone https://github.com/MischaRauch/serverless_machine_learning.git```
2. Navigate to the repository directory.
3. Install the required dependencies, including Hugging Face libraries:
```pip install -r requirements.txt```
## Datasets in 01_Assignment

### Flower Dataset

The Flower dataset, commonly used for classification tasks in machine learning, is featured in this part of the project. It typically consists of images of various flower species, making it ideal for testing image recognition algorithms.

#### Dataset Characteristics:
- **Type**: Image classification.
- **Classes**: Multiple, usually including species like roses, daisies, sunflowers, etc.
- **Usage**: Demonstrates the application of convolutional neural networks (CNNs) or other image processing techniques.

#### Implementation Details:
- Code snippets for loading and preprocessing the dataset.
- Description of the model architecture used for classification.
- Instructions for training and evaluating the model on this dataset.

### Wine Dataset

The Wine dataset is another classic dataset used for regression or classification tasks. It usually contains chemical analysis results of different wine samples, aiming to classify them into various types.

#### Dataset Characteristics:
- **Type**: Classification or regression.
- **Features**: Chemical properties like alcohol content, acidity, etc.
- **Target**: Wine type or quality rating.

#### Implementation Details:
- Steps for data preprocessing and feature selection.
- Description of the machine learning model used (e.g., decision trees, random forests).
- Instructions for model training, testing, and evaluation.

### Enviornment
Both datasets are hosted on Hopsworks.

## License
Apache 2.0.
