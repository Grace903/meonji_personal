# 기상청 날씨데이터 전처리 - kma_weather_ready.csv 파일 생성

import pandas as pd
import glob
import numpy as np

file_path = 'filtered_weather_yearly_*.csv'

csv_files = glob.glob(file_path)
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

df['YYMMDDHHMI'] = pd.to_datetime(df['YYMMDDHHMI'], format='%Y%m%d%H%M')
df = df.sort_values(by='YYMMDDHHMI', ascending=True)

# 날짜 관련 열 추가 함수
def timeshape(df):
    df['YYMMDDHHMI'] = pd.to_datetime(df['YYMMDDHHMI'], format='%Y%m%d%H%M')
    
    # 년, 월, 일, 시, 요일 열 추가
    df['year'] = df['YYMMDDHHMI'].dt.year
    df['month'] = df['YYMMDDHHMI'].dt.month
    df['day'] = df['YYMMDDHHMI'].dt.day
    df['hour'] = df['YYMMDDHHMI'].dt.hour
    df['weekday'] = df['YYMMDDHHMI'].dt.weekday  # 월요일=0, 일요일=6

    return df

# 날짜 관련 열 추가 후 반환
df = timeshape(df)

# 날짜 범위 생성 (2012년 1월 1일부터 2024년 12월 31일까지, 시간 간격 1시간)
date_range = pd.date_range(start='2012-01-01 00:00:00', end='2024-12-31 23:00:00', freq='h')
all_dates_df = pd.DataFrame(date_range, columns=['YYMMDDHHMI'])

# 이미 존재하는 df의 날짜 형식 'YYYY-MM-DD HH:MM:SS'로 맞춰주기 위해 변환
df['YYMMDDHHMI'] = pd.to_datetime(df['YYMMDDHHMI'], format='%Y-%m-%d %H:%M:%S')

# 누락된 날짜 찾기
no_date_df = all_dates_df[~all_dates_df['YYMMDDHHMI'].isin(df['YYMMDDHHMI'])]

# 누락된 날짜 데이터를 동일한 형식으로 변환
no_date_df['YYMMDDHHMI'] = pd.to_datetime(no_date_df['YYMMDDHHMI'], format='%Y-%m-%d %H:%M:%S')

# 누락된 날짜를 추가할 때 형식이 맞춰지게 됨
# print(no_date_df.head())





# 누락된 날짜와 시간 추가
def add_missing_data(df, no_date_df):
    # 누락된 데이터에 -9.0 값 채우기
    columns = df.columns[1:]  # 'YYMMDDHHMI'는 제외한 나머지 컬럼
    for index, row in no_date_df.iterrows():
        new_row = {'YYMMDDHHMI': row['YYMMDDHHMI']}  # 새로운 날짜값

        # 다른 열은 -9.0으로 채움
        for col in columns:
            new_row[col] = -9.0

        # df_all에 추가
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 'YYMMDDHHMI' 기준으로 다시 정렬
    df = df.sort_values(by='YYMMDDHHMI').reset_index(drop=True)
    return df

# 누락된 데이터를 df_all에 추가
df = add_missing_data(df, no_date_df)

# 결과 출력
print(df.head())








# 출력 확인
# print(df.head())

column_order = [
    'year', 'month', 'day', 'hour', 'weekday',
    'WD','WS','TA','TD','HM','RN','SD_TOT',
    'CA_TOT','CA_MID','VS','TS','SI','PS','PA'
]

df = df[column_order]

# df.to_csv('weather_data_all.csv', index=False) # df 파일 필요시 저장

weather_mean = ['WD','WS','HM','CA_MID','VS','PS','PA']
mean_values = df[weather_mean].apply(lambda x: x[x != -9.0].mean()) 
for col in weather_mean:
    df[col] = df[col].replace(-9.0, mean_values[col])

weather_zero = ['RN','SD_TOT']
for col in weather_zero:  
    df[col] = df[col].replace(-9.0, 0)

# CA_TOT 결측치 처리 (해당 월 평균값)
def fill_ca_tot(row):
    if row['CA_TOT'] == -9.0:
        mean_ca_tot = df[df['month'] == row['month']]['CA_TOT'].replace(-9.0, np.nan).mean()
        return mean_ca_tot
    else:
        return row['CA_TOT']

df['CA_TOT'] = df.apply(fill_ca_tot, axis=1)

def fill_si(row):
    # 하절기: 4월 ~ 9월, 동절기: 10월 ~ 3월
    summer = 4 <= row['month'] <= 9
    winter = not summer

    # 결측치 SI 처리
    if row['SI'] == -9.0:
        # 하절기: 6시~21시 결측치는 평균값으로 대체
        if summer and 6 <= row['hour'] < 21:
            mean_si = df[(df['hour'] >= 6) & (df['hour'] < 21) & (df['month'] == row['month'])]['SI'].replace(-9.0, np.nan).mean()
            return mean_si
        
        # 동절기: 8시~19시 결측치는 평균값으로 대체
        if winter and 8 <= row['hour'] < 19:
            mean_si = df[(df['hour'] >= 8) & (df['hour'] < 19) & (df['month'] == row['month'])]['SI'].replace(-9.0, np.nan).mean()
            return mean_si
        
        # 나머지 밤 시간대 결측치는 0으로 대체
        else:
            return 0
    
    # 결측치가 아니면 원래 값 유지
    return row['SI']

df['SI'] = df.apply(fill_si, axis=1)

df.columns = df.columns.str.lower()

# df_null = (df == -9.0).sum()
# print (df_null)
    
df.to_csv('kma_weather_ready.csv', index=False)

# print(df)