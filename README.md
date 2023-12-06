Open-Source-SW---Pandas-Task

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
