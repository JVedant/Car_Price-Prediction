import os
import numpy as np

import config

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble

def run(fold):
   df = pd.read_csv(config.TRAINING_FILE_ENCODED)

   df_train = df[df.kfold != fold].reset_index(drop=True)
   df_valid = df[df.kfold == fold].reset_index(drop=True)

   x_train = df_train.drop("selling_price", 1).values
   y_train = df_train.selling_price.values

   x_valid = df_valid.drop("selling_price", 1).values
   y_valid = df_valid.selling_price.values

   # hyper params tuning
   n_estimators = [int(x) for x in np.arange(100, 1500, 100)]
   max_features = ['auto', 'sqrt']
   max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
   min_samples_split = [2, 5, 10, 15, 100]
   min_samples_leaf = [1, 2, 5, 10]

   # Create the random grid
   random_grid = {
         'n_estimators': n_estimators,
         'max_features': max_features,
         'max_depth': max_depth,
         'min_samples_split': min_samples_split,
         'min_samples_leaf': min_samples_leaf
      }

   rf_random = model_selection.RandomizedSearchCV(
      estimator = ensemble.RandomForestRegressor(),
      param_distributions = random_grid,
      scoring = 'neg_mean_squared_error',
      n_iter = 10,
      cv = 5,
      verbose=1,
      random_state=42,
      n_jobs = 1)

   rf_random.fit(x_train,y_train)

   model = rf_random.best_estimator_

   model.fit(x_train, y_train)

   preds = model.predict(x_valid)

   mse = metrics.mean_squared_error(y_valid, preds)
   print(f"Fold = {fold}, mean_squared_error={mse}")

   joblib.dump(
      model,
      os.path.join(config.MODEL_OUTPUT, f"RandomForest/fold{fold}.pkl")
   )

if __name__ == "__main__":
   for fold_ in range(5):
      run(fold_)