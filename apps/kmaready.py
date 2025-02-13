# 기상청 날씨데이터 전처리 - kma_weather_ready.csv 파일 생성

import pandas as pd
import glob
import numpy as np

file_path = 'data/filtered_weather_yearly_*.csv'

csv_files = glob.glob(file_path)
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# 날짜 관련 열 추가 함수
def timeshape(df):
    df['YYMMDDHHMI'] = pd.to_datetime(df['YYMMDDHHMI'], format='%Y%m%d%H%M')
    
    # 년, 월, 일, 시, 요일 열 추가
    df['year'] = df['YYMMDDHHMI'].dt.year
    df['month'] = df['YYMMDDHHMI'].dt.month
    df['day'] = df['YYMMDDHHMI'].dt.day
    df['hour'] = df['YYMMDDHHMI'].dt.hour
    df['weekday'] = df['YYMMDDHHMI'].dt.weekday  # 월요일=0, 일요일=6

# 날짜 관련 열 추가
timeshape(df)

print(type(df['YYMMDDHHMI']))

# 날짜 범위 생성 (2012년 1월 1일부터 2024년 12월 31일까지, 시간 간격 1시간)
date_range = pd.date_range(start='2012-01-01 00:00', end='2024-12-31 23:00', freq='h')
all_dates_df = pd.DataFrame(date_range, columns=['YYMMDDHHMI'])
all_dates_df['YYMMDDHHMI'] = all_dates_df['YYMMDDHHMI'].dt.strftime('%Y%m%d%H%M')

# df에서 누락된 날짜 찾기
df['YYMMDDHHMI'] = df['YYMMDDHHMI'].astype(str)
no_date_df = all_dates_df[~all_dates_df['YYMMDDHHMI'].isin(df['YYMMDDHHMI'])]

# print(all_dates_df.head())

# no_date_df = pd.DataFrame(df['YYMMDDHHMI'])

# 날짜 관련 열 추가
timeshape(no_date_df)
print(type(no_date_df['YYMMDDHHMI']))

# 기상 데이터 열에 결측값 채우기 위한 초기값 추가
columns_to_add = ['WD', 'WS', 'HM', 'RN', 'SD_TOT', 'CA_TOT', 'CA_MID', 'VS', 'SI', 'PS', 'PA', 'TA', 'TD', 'TS']
for col in columns_to_add:
    no_date_df[col] = -9.0

# 온도 관련 컬럼 : ['TA', 'TD', 'TS',]

df = pd.concat([df, no_date_df], ignore_index=True)
# df = df.sort_values(by='YYMMDDHHMI').reset_index(drop=True)

df['YYMMDDHHMI'] = pd.to_datetime(df['YYMMDDHHMI'], format='%Y%m%d%H%M')  # YYMMDDHHMI를 날짜형으로 변환
df = df.sort_values(by='YYMMDDHHMI').reset_index(drop=True)

print(df.head())

# 없는 날짜 채우기 위한 초기값 설정
def fill_value(df, no_date_df):
    for no_time in no_date_df['YYMMDDHHMI']:
        no_time = pd.to_datetime(str(no_time), format='%Y%m%d%H%M')
        
        # 해당 날짜가 df에 없다면
        if no_time not in df['YYMMDDHHMI'].values:
            fill_data = {'YYMMDDHHMI': no_time}
            
            # 모든 열에 대해 기본값(-9.0)을 채운다.
            for col in df.columns:
                if col != 'YYMMDDHHMI':  # YYMMDDHHMI는 제외
                    fill_data[col] = -9.0
            
            # 새로운 행 추가
            df = pd.concat([df, pd.DataFrame([fill_data])], ignore_index=True)

    # 데이터프레임을 YYMMDDHHMI 기준으로 정렬
    return df.sort_values(by='YYMMDDHHMI').reset_index(drop=True)

# 없는 날짜 채우기
df = fill_value(df, no_date_df)

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


# 기존의 df에서 날짜 관련 열 삭제
df = df.drop(columns=['YYMMDDHHMI'])

column_order = [
    'year', 'month', 'day', 'hour', 'weekday',
    'WD','WS','TA','TD','HM','RN','SD_TOT',
    'CA_TOT','CA_MID','VS','TS','SI','PS','PA'
]

df = df[column_order]

df.columns = df.columns.str.lower()

# df_null = (df == -9.0).sum()
# print (df_null)
    
# df.to_csv('kma_weather_ready.csv', index=False)

# print(df)

# 2023년 12월 10일 23시에 해당하는 데이터 필터링
# filtered_data = df[(df['year'] == 2023) & 
#                        (df['month'] == 12) & 
#                        (df['day'] == 10) & 
#                        (df['hour'] == 23)]

# # 필터링된 데이터 출력
# print(filtered_data)
