# Car_Price-Prediction

An app made using Flask to predict the car price using details like year of purchase, purchasing price, number of owners, etc.

---
Technique used: 
  - Decision tree Regressor
  - Random Forest Regressor
  
---  
the template used can be found on [index.py](https://github.com/JVedant/Car_Price-Prediction/blob/master/templates/index.html)

---
the dataset used here is very small and can be found at [car data.csv](https://github.com/JVedant/Car_Price-Prediction/blob/master/input/car%20data.csv)

---
How to run:
1. run [create_folds.py](https://github.com/JVedant/Car_Price-Prediction/blob/master/src/create_folds.py) from src to create folds in the dataset.
2. run [encode_data.py](https://github.com/JVedant/Car_Price-Prediction/blob/master/src/encode_data.py) from src to encode the data using pandas dummmy method and to select the features.
3. run [model.py](https://github.com/JVedant/Car_Price-Prediction/blob/master/src/model.py) to train the model and dump it using joblib at [model.pkl](https://github.com/JVedant/Car_Price-Prediction/blob/master/models/DecisionTree/model.pkl)
    - you can check the [model.pkl](https://github.com/JVedant/Car_Price-Prediction/blob/master/models/model.pkl) also, it is trained on random forest regressor and is shown in [EDA.ipynb](https://github.com/JVedant/Car_Price-Prediction/blob/master/notebooks/EDA.ipynb)
4. run app.py
