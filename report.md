# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Jahnvi Sahni

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
The model did not perform as well as predicted the first time I utilized the raw dataset without performing any data analysis or feature engineering because it had a lot of errors. I had to change the negative values with 0 in order to be able to submit my findings to Kaggle. 

### What was the top ranked model that performed?
The  WeightedEnsemble_L3 model that used the data with created features 

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
For the extra features I divided the datetime in month, day, year and hour. Also it was usefull to transform the season and weather features to categorical

### How much better did your model preform after adding additional features and why do you think that is?
Because additional features can be good predictors to estimate the target value, in this case I decided to separate the date becuase it helps the model to analyse seasonality paterns in the data which can be usefull for a regression model

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
While hyperparameter adjustment was helpful in certain situations, it did not significantly increase model performance. Certain configurations were helpful, but others negatively affected the model's performance.

### If you were given more time with this dataset, where do you think you would spend more time?
To learn more about this dataset, conduct a more thorough data analysis and further investigate the hiperparameters.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|default_vals|default_vals|default_vals|1.80509|
|add_features|default_vals|default_vals|default_vals|0.47022|
|hpo|num_leaves: lower=26, upper=66|dropout_prob: 0.1, 1.0|num_boost_round: 100|0.46014|



### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![plot1.PNG]()
