# -*- coding: utf-8 -*-
"""OSS_project2_2_code_template

Inha University Open source SW introductory course assignment

Original file is located at
    https://colab.research.google.com/drive/1Mxxo80m8bgIayjVnhKtJRC97qkoHrii1

#2-2 프로젝트 목표

특정 연도의 타자의 연봉을 예측하기 위해 다양한 ML 모델을 훈련시킵니다.

이것은 회귀이며 3가지 종류의 ML 모델을 사용합니다.

1. Decision Tree Regressor
2. Random Forest Regressor
3. Support Vector Machine Regressor

(We will use only numerical features)

# 프로젝트 요구사항

1. 전체 데이터를 연도(해당 시즌) 열 기준으로 오름차순 정렬.
2. 전체 데이터를 학습/테스트 데이터세트로 분할합니다.
3. 숫자열만 추출
▪ 숫자 열: 'age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI','SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war'
4. 의사 결정 트리, 랜덤 포레스트 및 svm에 대한 학습 및 예측 기능을 완료합니다.
5. 주어진 라벨 및 예측에 대한 RMSE 계산.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

def sort_dataset(dataset_df):
  return dataset_df.sort_values(by='year', ascending=True)

def split_dataset(dataset_df):
  X = dataset_df.drop('salary', axis=1)
  Y = dataset_df['salary'] * 0.001
  # 인덱스 길이가 1913이고 train set에 1718길이 만큼 넣고 싶으니 나누면 약 0.8980658이다. -> test_size에 1-0.8980658 = 약0.1019를 넣는다.
  return train_test_split(X, Y, test_size=0.1019, random_state=0, shuffle=False)

def extract_numerical_cols(dataset_df):
  return dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]

def train_predict_decision_tree(X_train, Y_train, X_test):
  dtr = DecisionTreeRegressor()
  dtr.fit(X_train, Y_train)
  return dtr.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):
  rfr = RandomForestRegressor()
  rfr.fit(X_train, Y_train)
  return rfr.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
  pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVR())])
  pipeline.fit(X_train, Y_train)
  return pipeline.predict(X_test)

def calculate_RMSE(labels, predictions):
  mse = mean_squared_error(labels, predictions)
  return np.sqrt(mse)

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

	sorted_df = sort_dataset(data_df)
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)

	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
