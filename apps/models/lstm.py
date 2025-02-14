import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler


df = pd.read_csv('data/finedust_basic_data_12.csv')


    # 계절 정보 추가 (season: 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울)
def season_col(df):
    df['season'] = np.select(
        [
            (df['month'].isin([3, 4, 5])),  # 봄 (3, 4, 5월)
            (df['month'].isin([6, 7, 8])),  # 여름 (6, 7, 8월)
            (df['month'].isin([9, 10, 11])),  # 가을 (9, 10, 11월)
            (df['month'].isin([12, 1, 2]))  # 겨울 (12, 1, 2월)
        ],
        [1, 2, 3, 4],  # 각각 봄, 여름, 가을, 겨울을 1, 2, 3, 4로 할당
        default=0  # 기본값은 0 (예외 처리)
    )
    # 계절 원핫 인코딩
    season_encoded = pd.get_dummies(df['season'], prefix='season', drop_first=False)
    df = pd.concat([df, season_encoded], axis=1)
    df = df.drop(columns=['season'])
    
    return df, season_encoded


def dayOfTheWeek_col(df):
    # 평일/주말 구분 컬럼 추가
    df['weekend'] = df['week'].isin([5, 6]).astype(int)  # 주말
    df['weekday'] = df['week'].isin([0, 1, 2, 3, 4]).astype(int)  # 평일

    # 요일 원핫인코딩
    week_encoded = pd.get_dummies(df['week'], prefix='week', drop_first=False)  # 요일 원핫인코딩
    df = pd.concat([df, week_encoded], axis=1)
    df = df.drop(columns=['week'])
    
    return df, week_encoded


# 계절 및 요일 컬럼 추가
df, season_encoded = season_col(df)
df, week_encoded = dayOfTheWeek_col(df)

# 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)



# 원인 데이터
def input_data(scaled_df, week_encoded, season_encoded):
    df_latlon = scaled_df[['lat', 'lon']]
    df_time = scaled_df[['year', 'month', 'day', 'hour'] + list(week_encoded.columns) + list(season_encoded.columns)]
    df_climate = scaled_df[['wd', 'ws', 'ta', 'td', 'hm', 'rn', 'sd_tot', 'ca_tot', 'ca_mid', 'vs', 'ts', 'si', 'ps', 'pa']]
    
    # x는 입력 변수들, y는 결과 변수
    x = pd.concat([df_latlon, df_time, df_climate], axis=1)  # x에는 lat, lon, 시간, 날씨 등의 정보
    y = scaled_df[['pm10', 'pm25', 'no2', 'o3', 'co', 'so2']]  # y는 미세먼지 농도 (결과)
    
    return x, y


def sliding_window_split(x, y, window_size, step_size):
    x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
    
    # 데이터 슬라이딩 윈도우로 분할
    for start in range(0, len(x) - window_size, step_size):
        end = start + window_size
        # 훈련 데이터
        x_train_list.append(x[start:end])
        y_train_list.append(y[start:end])
        # 테스트 데이터 (훈련 데이터 뒤의 일정 구간)
        x_test_list.append(x[end:end + window_size])
        y_test_list.append(y[end:end + window_size])

    # 리스트를 numpy array로 변환하여 반환
    return (np.array(x_train_list), np.array(y_train_list), np.array(x_test_list), np.array(y_test_list))


# 데이터 준비
scaled_df, season_encoded = season_col(scaled_df)  # season 컬럼 생성 및 원핫인코딩
scaled_df, week_encoded = dayOfTheWeek_col(scaled_df)  # week 컬럼 생성 및 원핫인코딩
x, y = input_data(scaled_df, week_encoded, season_encoded)


# 슬라이딩 윈도우 적용
window_size = 100  # 훈련 데이터의 크기 (예: 100시간)
step_size = 24  # 이동 간격 (예: 하루마다)
x_train, y_train, x_test, y_test = sliding_window_split(x.values, y.values, window_size, step_size)


# 데이터 확인
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

    














# def sequence_data(data, sequence_length=[24, 168]):
#     x, y = [], []
    
#     for sequence in sequence_length:
#         for i in range(len(data) - sequence):
#             # x 데이터와 y 데이터를 numpy로 변환
#             x_temp = data[i:i+sequence]
#             y_temp = data[i+sequence]
            
#             x.append(x_temp)
#             y.append(y_temp)

#     # x, y를 numpy array로 반환
#     x_pollutants = np.concatenate(x, axis=0)
#     y_pollutants = np.concatenate(y, axis=0)

#     return x_pollutants, y_pollutants


# Pollutants 데이터 시퀀스 생성
# x_pollutants, y_pollutants = sequence_data(df_pm.values, sequence_length=[24, 168])

# 데이터 나누기
# x_train, x_test, y_train, y_test = train_test_split(x_pollutants, y_pollutants, test_size=0.1, shuffle=False)


# R² 메트릭 정의
def r2(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())




# LSTM 모델 정의
model = Sequential()
# tanh (하이퍼볼릭 탄젠트) : 출력 범위가 -1과 1 사이, 음수 값을 포함하는 데이터 처리 시 유리함
# input_shape=(X_train.shape[1], X_train.shape[2]): 입력 데이터의 형태를 지정
# X_train.shape에서 1은 시퀀스 길이 (시간 단계), 2는 피처의 수(각 시점에서의 변수 수)
model.add(LSTM(64, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh'))
model.add(Dense(6))
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae', 'mse', r2])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# LearningRateScheduler 콜백 정의
def scheduler(epoch, lr):
    if epoch > 10:
        lr = lr * 0.9
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

# 모델 훈련
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
    )

# 학습 과정 시각화
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

# 모델 평가
test_loss = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')

# 예측 결과 시각화
y_pred = model.predict(x_test)

mse_train = mean_squared_error(y_train, model.predict(x_train))
mse_test = mean_squared_error(y_test, y_pred)
print(f"Train MSE: {mse_train}, Test MSE: {mse_test}")

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')


# 예측값과 실제값 비교 (여기서는 PM10과 PM25 예시)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(y_test[:50, 0], label='True PM10')  # 실제 PM10 값
plt.plot(y_pred[:50, 0], label='LSTM PM10')  # 예측 PM10 값
plt.legend()
plt.title('True vs LSTM PM10')

plt.subplot(1, 2, 2)
plt.plot(y_test[:50, 1], label='True PM25')  # 실제 PM25 값
plt.plot(y_pred[:50, 1], label='LSTM PM25')  # 예측 PM25 값
plt.legend()
plt.title('True vs LSTM PM25')

plt.show()




# 모델 저장
model_path = '/models/model/pm_lstm01.h5'

# 경로가 없다면 디렉토리 생성
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save(model_path)