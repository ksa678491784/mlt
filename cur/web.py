#!/usr/bin/env python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

st.header('Датасет')
main_status = st.text('')
read_state = st.text('Чтение датасета...')

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

read_state.text('Датасет загружен!')

test_size = st.sidebar.slider("test_size", 0.1, 0.9, value = 0.3)
random_state = st.sidebar.slider("random_state", 1, 100, value=10)
data_size = st.sidebar.slider("Data size", 100, orig_data.shape[0], value=200)
logistic_regression_c = st.sidebar.slider("Logistic regression C", 0.01, 1.0, value=1.0)
knn_n_neighbors = st.sidebar.slider("KNN n_neighbors", 2, 30, value=15)
decision_tree_max_depth = st.sidebar.slider("Decision tree max_depth", 3, 20, value=7)
random_forest_estimatiors = st.sidebar.slider("Random forest n_estimators", 1, 1400, value=600)
gradient_boost_estimators = st.sidebar.slider("Gradient boost n_estimators", 1, 700, value=400)

data = orig_data[:data_size]

st.subheader('head:')
st.write(data.head())

target_option = "satisfaction"
feature_cols = []
st.sidebar.subheader('Признаки:')
for col in data.columns:
  if col != target_option:
    cb = st.sidebar.checkbox(col, value=True)
    if cb:
      feature_cols.append(col)

scaler = st.sidebar.radio("Мастабирование", ("Нет", "MinMaxScaler", "StandardScaler"))
if scaler == 'MinMaxScaler':
  sc2 = MinMaxScaler()

  for col in data.columns:
    if col != target_option:
      data[col] = sc2.fit_transform(data[[col]])

  st.subheader('Мастабирование:')
  st.write(data.head())
elif scaler == 'StandardScaler':
  sc2 = StandardScaler()

  for col in data.columns:
    if col != target_option:
      data[col] = sc2.fit_transform(data[[col]])

  st.subheader('Мастабирование:')
  st.write(data.head())

main_status.text('В процессе обучения...')
data_X = data.loc[:, feature_cols]
data_Y = data.loc[:, target_option]
data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(
  data_X,
  data_Y,
  test_size=test_size,
  random_state=1,
)

lr = LogisticRegression(n_jobs=-1, C=logistic_regression_c)
lr.fit(data_X_train, data_y_train)

knc = KNeighborsClassifier(n_jobs=-1, n_neighbors=knn_n_neighbors)
knc.fit(data_X_train, data_y_train)

dtc = DecisionTreeClassifier(max_depth=decision_tree_max_depth)
dtc.fit(data_X_train, data_y_train)

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=random_forest_estimatiors)
rfc.fit(data_X_train, data_y_train)

gbc = GradientBoostingClassifier(n_estimators=gradient_boost_estimators)
gbc.fit(data_X_train, data_y_train)

main_status.text("Обучено!")

metrics = [precision_score, recall_score, roc_auc_score]
metr = [i.__name__ for i in metrics]
metrics_ms = st.sidebar.multiselect("Метрики", metr)

methods = [lr, knc, dtc, rfc, gbc]
md = [i.__class__.__name__ for i in methods]
methods_ms = st.sidebar.multiselect("Методы обучения", md)

selMethods = []
for i in methods_ms:
  for j in methods:
    if i == j.__class__.__name__:
      selMethods.append(j)

selMetrics = []
for i in metrics_ms:
  for j in metrics:
    if i == j.__name__:
      selMetrics.append(j)

st.header('Оценка')
for name in selMetrics:
  st.subheader(name.__name__)

  array_labels = [ ]
  array_metric = [ ]

  for func in selMethods:
    y_pred = func.predict(data_X_test)

    array_labels.append(func.__class__.__name__)
    array_metric.append(name(y_pred, data_y_test))

    st.text("{} - {}".format(func.__class__.__name__, name(y_pred, data_y_test)))

  fig, ax1 = plt.subplots(figsize=(3,3))
  pos = np.arange(len(array_metric))
  rects = ax1.barh(
    pos,
    array_metric,
    align="center",
    height=0.5,
    tick_label=array_labels,
  )
  for a, b in zip(pos, array_metric):
    plt.text(0, a - 0.1, str(round(b, 3)), color="white")
  st.pyplot(fig)
