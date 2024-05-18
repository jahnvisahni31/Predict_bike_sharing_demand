# Predict Bike Sharing with AutoGluon in AWS SageMaker Studio

This repository contains a Jupyter Notebook (`.ipynb` file) that demonstrates how to use AutoGluon to predict bike sharing demand in AWS SageMaker Studio. AutoGluon is a powerful AutoML toolkit that automates the process of training and tuning machine learning models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [links](#links)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [AutoGluon Workflow](#autogluon-workflow)
- [Evaluation](#evaluation)
- [Cleanup](#cleanup)
- [References](#references)

## Prerequisites

Before you begin, ensure you have the following:

- An AWS account with access to SageMaker Studio.
- SageMaker Studio set up in your AWS environment.
- Basic knowledge of Jupyter Notebooks and Python.

## Links
[Open in Google Colab](https://colab.research.google.com/github/jahnvisahni31/predict_bike_sharing_with_autogluon/blob/main/predict_bike_sharing_with_autogluon.ipynb)


## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/jahnvisahni31/predict_bike_sharing_with_autogluon.git
    cd predict_bike_sharing_with_autogluon
    ```

2. **Open SageMaker Studio:**

    Launch SageMaker Studio from the AWS Management Console.

3. **Upload the Notebook:**

    Upload the `predict_bike_sharing_with_autogluon.ipynb` file to your SageMaker Studio environment.

4. **Install Required Libraries:**

    Open a terminal in SageMaker Studio and run the following command to install AutoGluon:

    ```bash
    pip install autogluon
    ```

## Dataset

The dataset used in this example is the [Bike Sharing Demand dataset](https://www.kaggle.com/c/bike-sharing-demand) from Kaggle. You can download the dataset and upload it to your SageMaker Studio environment.

## Usage

1. **Open the Notebook:**

    Open the `predict_bike_sharing_with_autogluon.ipynb` file in SageMaker Studio.

2. **Follow the Steps:**

    Follow the steps in the notebook to:

    - Load the dataset.
    - Preprocess the data.
    - Train the model using AutoGluon.
    - Evaluate the model's performance.

## AutoGluon Workflow

The notebook demonstrates the following AutoGluon workflow:

1. **Import Libraries:**
    ```python
    from autogluon.tabular import TabularPredictor
    ```

2. **Load Dataset:**
    ```python
    import pandas as pd
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    ```

3. **Train Model:**
    ```python
    predictor = TabularPredictor(label='count').fit(train_data)
    ```

4. **Evaluate Model:**
    ```python
    performance = predictor.evaluate(test_data)
    print(performance)
    ```

## Evaluation

The notebook includes steps to evaluate the trained model on a test set, providing metrics such as RMSE (Root Mean Squared Error) to measure the model's performance.

## Cleanup

After completing the notebook, remember to clean up any resources to avoid unnecessary charges:

- Delete any endpoints or instances created during the process.
- Remove datasets and notebooks from your SageMaker Studio environment if no longer needed.

## References

- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [AWS SageMaker Studio Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html)
- [Bike Sharing Demand Dataset on Kaggle](https://www.kaggle.com/c/bike-sharing-demand)

---

This README provides a high-level overview of using AutoGluon for bike sharing prediction in AWS SageMaker Studio. For detailed instructions and code, please refer to the included Jupyter Notebook.
