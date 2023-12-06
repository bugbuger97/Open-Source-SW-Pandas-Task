Open-Source-SW---Pandas-Task

과제 2-1

import pandas as pd

file_path = './2019_kbo_for_kaggle_v2.csv'

bb = pd.read_csv(file_path) # bb == baseball의 약자

문제 2-1-1. H, avg, HR, OBP의 TOP 10 player 출력

for i in range(2015,2019):
  
  bb_df = bb[(bb['year'] == i)]
  
  H = bb_df.sort_values(by='H', ascending=False).head(10)['batter_name'].tolist()
  
  avg = bb_df.sort_values(by='avg', ascending=False).head(10)['batter_name'].tolist()
  
  HR = bb_df.sort_values(by='HR', ascending=False).head(10)['batter_name'].tolist()
  
  OBP = bb_df.sort_values(by='OBP', ascending=False).head(10)['batter_name'].tolist()
  
  result = pd.DataFrame({'H' : H,'avg' : avg,'HR' : HR,'OBP' : OBP}, index = range(1, 11))
  
  print(f'{i} year Top10 player\n')
  
  print(result)
  
  print('\n')

문제 2-1-2. war (승리 기여도)에 따른 가장 높은 값의 player 출력 by position (cp) in 2018. (15 points)

데이터 프레임의 열들을 이렇게 나열 - 포수, 1루수, 2루수, 3루수, 유격수, 좌익수, 중견수, 우익수

Players with a high 'war' category are selected to the top 30 and put into a list according to 'cp' and printed.

bb_df = bb[(bb['year'] == 2018)]

temp = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

value=[] # DataFrame의 batter_name list

index=[] # DataFrame의 cp list

war = bb_df.sort_values(by='war', ascending=False).head(30)['batter_name'].tolist()

cp = bb_df.sort_values(by='war', ascending=False).head(30)['cp'].tolist()

combine = list(zip(cp,war)) # war를 기준으로 내림차순하여 30위까지의 정보를 끌어와서 (cp,batter_name)으로 묶는다

for i,j in combine:

  if len(temp) != 0:
  
    if i in temp:
    
      index.append(i)
      
      value.append(j)
    
      temp.remove(i)
  
  else:
    
    break

result = pd.DataFrame({'war':value}, index = index)

result

문제 2-1-3. R(득점), H(안타), HR(홈런), RBI(타점), SB(도루), war(승리 기여도), avg(타율), OBP 중(출루율), SLG(장타율) 중 연봉(연봉)과 가장 상관관계가 높은 것은?

print(bb.corr()['salary'].sort_values(ascending=False)) # salary에 대한 상관관계 내림차순

결과를 보시면 salary에 대한 상관관계에서 가장 높은 값은 RBI(타점)에 있는 것을 볼 수 있다.

result = bb.corr()['salary'].sort_values(ascending=False)[1]

print(f'\nsalary와의 상관관계가 RBI가 {result}로 가장 높은 것을 알 수 있다.\n')

------------------------------------------------------------------------------------------------------------------------------------------------------------------

과제 2-2

2-2 프로젝트 목표

특정 연도의 타자의 연봉을 예측하기 위해 다양한 ML 모델을 훈련시킵니다.

이것은 회귀이며 3가지 종류의 ML 모델을 사용합니다.

1. Decision Tree Regressor
2. Random Forest Regressor
3. Support Vector Machine Regressor

(We will use only numerical features)

프로젝트 요구사항

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
  인덱스 길이가 1913이고 train set에 1718길이 만큼 넣고 싶으니 나누면 약 0.8980658이다. -> test_size에 1-0.8980658 = 약0.1019를 넣는다.
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
	DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	
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
