# ANN-based-Banking-Churn-Prediction
- This repository will have complete machine learning and deep learning based banking churn prediction ANN model which will analyze tha probablity for a customer to leave.
- The project was deployed on Google Cloud Platform as well as completely tested on Localhost.
</br>

## Project Description
### Welcome Screen
- The main welcome screen is made in **HTML5** and **CSS3** with a basic and simple design.
- Here is the main screen for the **Bank Churn Prediction** Interface for the bank admin.</br>

![Welcome Screen](https://github.com/paras009/ANN-based-Banking-Churn-Prediction/blob/master/images/3welcome_screen.PNG)

### Bank Admin Input
- The bank employee has to enter the details of the customer whose churn they want to analyze.
- Below is the screenshot of the input being filled by bank employee.</br>

![Filled Screen](https://github.com/paras009/ANN-based-Banking-Churn-Prediction/blob/master/images/4filled_index.png)

## Analysis and Accuracy
- The Prediction engine is built over a deep **Artificial Neural Network** backed with **[Keras](https://www.tensorflow.org/guide/keras)**.
- I have achieved an accuracy of around **~85%** on both training and testing data.</br>
- The ANN is trained over K-fold cross validation testing over 10 rounds to find if it was underfit or overfit over the data based on the variance betweent the accuracies of the 10 rotations.
- The model is Tuned over the Hyerparametes to find the best **batch_size**, **epoch** and **optimizer** for generating the best possible combination for best fit model.

![Accuracy Python Console](https://github.com/paras009/ANN-based-Banking-Churn-Prediction/blob/master/images/1accuracy_console.PNG)

## Deployment and Production
-  The **API interfacing** for the deplyment on [Localhost](http://localhost:8080/index) is done using [Flask](https://flask.palletsprojects.com/en/1.1.x/).
- The server is run on Local system during the staging of the project.
- Older deployment was done on [Google Cloud Platform](https://cloud.google.com/)
- Recently, the final deployment was done on **Heloku** platform and can be accessed from the link below.
- LINK: [https://banking-churn-pediction](https://banking-churn-pediction.herokuapp.com)

## Predictions
- The final prediction of the model is the percentage of churn for that customer.
-  The prediction signifies the chances of the customer to leave the services of the bank which makes the bank to _focus more on such such customers_ and try to retain them using **[Sales and Marketing strategies](https://github.com/paras009/Sales-and-Marketing-Analytics)** about which I have worked in this [GitHub](https://github.com/paras009/Sales-and-Marketing-Analytics) module.</br>

![Prediction Screenshot](https://github.com/paras009/ANN-based-Banking-Churn-Prediction/blob/master/images/5prediction.PNG)

## Contribution
- The project is built completely by Paras Varshney.</br>
Connect on [LinkedIn](https://www.linkedin.com/in/pv009)</br>
Follow on [Medium](https://medium.com/@pv009)</br>
Follow on [Github](https://github.com/paras009)</br>

#### Thank You!
