I have did this project as a part of FreeCodeCamp Machine Learning with Python course. 

In this challenge, I have predicted healthcare costs using a regression algorithm.

A dataset was given that contains information about different people including their healthcare costs. I have used  the data to predict healthcare costs based on new data.

The first two cells of this notebook import libraries and the data.

Then I made sure to convert categorical data to numbers. And used 80% of the data as the train_dataset and 20% of the data as the test_dataset.

Then as asked in the challenge I popped  off the "expenses" column from these datasets to create new datasets called train_labels and test_labels. And I have used these labels when training your model.

Now as next part of the challenge I have created a model and trained it with the train_dataset. And ran the final cell in this notebook to check your model. The final cell will use the unseen test_dataset to check how well the model generalizes.

To pass the challenge, model.evaluate must return a Mean Absolute Error of under 3500. This means it predicts health care costs correctly within $3500.
