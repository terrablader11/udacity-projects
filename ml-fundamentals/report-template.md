# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Zachary Li

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
I had to add the datetime to use as an index as well as remove the original index. This was accomplished by creating a DataFrame with all the datetime values, and then adding the predictions onto the DataFrame. 

### What was the top ranked model that performed?
For all the attempts, the WeightedEnsemble performed best. Generally, the L3 model outperformed the L2 version, but not always. 

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The exploratory data analysis showed that many of the features had strong correlations with teh target but also revealed that the datetime and cateogrical modules could become more useful. Therefore, we made several of the 0/1 features in categorical variables and split the datetime into an hour feature. 

### How much better did your model preform after adding additional features and why do you think that is?
It performed substanially better, probably because the time information is highly useful in predicting bike sharing. No one will be using a bikeshare during the night, but during the day, the number will be much higher.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
It performed significantly better, but not nearly as much as the difference between features and hyperparameters. 

### If you were given more time with this dataset, where do you think you would spend more time?
I believe that more training time, as well as splitting the date time further could have improved the model. 

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

|model|time_limit|presets|hyperparameters|score|
|--|--|--|--|--|
|initial|600|best_quality|default|1.79748|
|add_features|600|best_quality|deafult|0.66655|
|hpo|900|best_quality|light|0.4589|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
In this project, we experimented with a Bike Sharing Dataset and examined how adding features as well as tuning hyperparameters improved the perofrmance of the model. The differences were highly apparent and showed the importance of each step of the ML development process. 