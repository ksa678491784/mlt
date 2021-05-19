#!/usr/bin/env python
import pandas as pd
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score

target = "satisfaction"

orig_data = pd.read_csv("satisfaction.csv").iloc[: , 1:]
orig_data.drop_duplicates()
orig_data.dropna(axis=0, how='any', inplace=True)

to_encode = [
  "Gender",
  "Customer Type",
  "Type of Travel",
  "Class",
]

for col in to_encode:
  orig_data[col] = LabelEncoder().fit_transform(orig_data[col])

orig_data["satisfaction"].replace("satisfied", 1, inplace=True)
orig_data["satisfaction"].replace("neutral or dissatisfied", 0, inplace=True)

scaler = StandardScaler()
for col in orig_data.columns:
  if col != target:
    orig_data[col] = scaler.fit_transform(orig_data[[col]])

data_X = orig_data.loc[:, [x for x in orig_data.columns if x != target]]
data_Y = orig_data.loc[:, target]
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(
  data_X,
  data_Y,
  test_size=0.3,
  random_state=1
)

cls = AutoSklearn2Classifier(time_left_for_this_task=60)
print("Fitting..")
cls.fit(data_X_train, data_y_train)
pred = cls.predict(data_X_test)
print("AutoML precision_score", precision_score(data_y_test, pred))
print("AutoML recall_score", recall_score(data_y_test, pred))
print("AutoML roc_auc_score", roc_auc_score(data_y_test, pred))
