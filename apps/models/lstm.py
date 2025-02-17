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
    # df = df.drop(columns=['season'])
    
    return df, season_encoded


def dayOfTheWeek_col(df):
    df['weekend'] = df['week'].isin([5, 6]).astype(int)  # 주말
    df['weekday'] = df['week'].isin([0, 1, 2, 3, 4]).astype(int)  # 평일

    # 요일 원핫인코딩
    week_encoded = pd.get_dummies(df['week'], prefix='week', drop_first=False)  # 요일 원핫인코딩
    df = pd.concat([df, week_encoded], axis=1)
    # df = df.drop(columns=['week'])
    
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


# 데이터 분리
df_latlon = scaled_df[['lat','lon']].values
df_time = scaled_df[['year','month','day','hour','season','weekend']].values  # 계절과 주말 정보 추가
df_climate = scaled_df[['wd','ws','ta','td','hm','rn','sd_tot','ca_tot','ca_mid','vs','ts','si','ps','pa']].values
df_pm = scaled_df[['pm10', 'pm25', 'no2', 'o3', 'co', 'so2']].values


x, y = input_data(scaled_df, week_encoded, season_encoded)



# x_train을 3D 형식인 (samples, timesteps, features)로 변환
# 그런데 (samples, features) 형태로 반환되고 있음 -> LSTM에 넣을 수 있도록 timesteps 차원을 추가
# 변환하려면 x_train, x_test를 numpy 배열로 변환 후 형상 변환을 추가

# def sliding_window_split(x, y, window_size, step_size):
#     x_train, y_train, x_test, y_test = [], [], [], []   # list로 초기화
    
#     # 데이터 슬라이딩 윈도우로 분할
#     for start in range(0, len(x) - window_size, step_size):
#         end = start + window_size
#         # print(f"start: {start}, end: {end}")
    
#         # 훈련 데이터
#         x_train.append(x.iloc[start:end, :].values) 
#         y_train.append(y.iloc[start:end].values)  

        
#     # print(f"x_train.append({x.iloc[start:end, :].values}.shape)")
#     # print(f"y_train.append({y.iloc[start:end].values}.shape)")
    
#     # 리스트를 numpy 배열로 변환
#     x_train = np.array(x_train)
#     y_train = np.array(y_train)
    
#     x_test = np.array(x_test)
#     y_test = np.array(y_test)
    
    
def sliding_window_split(x, y, window_size, step_size):
    x_train, y_train, x_test, y_test = [], [], [], []   # list로 초기화

    # 슬라이딩 윈도우로 훈련 데이터 분할
    for start in range(0, len(x) - window_size+1, step_size):
        end = start + window_size
        x_train.append(x.iloc[start:end, :].values)  # (window_size, features)
        y_train.append(y.iloc[start:end, :].values)  # (window_size, targets)

    # 슬라이딩 윈도우로 테스트 데이터 분할
    for start in range(len(x) - window_size, len(x), step_size):
        end = start + window_size
        x_test.append(x.iloc[start:end, :].values)  # (window_size, features)
        y_test.append(y.iloc[start:end, :].values)  # (window_size, targets)

    # 리스트를 numpy 배열로 변환
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # x_train과 x_test 형태 확인
    print(f"Before reshape, x_train shape: {x_train.shape}")
    print(f"Before reshape, y_train shape: {y_train.shape}")
    
    # y_train과 y_test는 2D에서 3D로 변환 (samples, window_size, targets)
    y_train = y_train.reshape((y_train.shape[0], window_size, y_train.shape[1]))  # (samples, window_size, targets)
    y_test = y_test.reshape((y_test.shape[0], window_size, y_test.shape[1]))  # (samples, window_size, targets)

    # 변환 후 x_train, y_train, x_test, y_test 형태 확인
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test


# 슬라이딩 윈도우 분할
window_size = 168  # 1주치 훈련 데이터
step_size = 24  # 이동 간격 (예: 하루마다)

# x.values와 y.values를 사용하여 슬라이딩 윈도우로 분할
x_train, y_train, x_test, y_test = sliding_window_split(x, y, window_size, step_size)




# R² 메트릭 정의
def r2(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


# LearningRateScheduler 콜백 정의
def scheduler(epoch, lr):
    if epoch > 10:
        lr = lr * 0.9
    return lr

lr_scheduler = LearningRateScheduler(scheduler)



# # x_train과 x_test를 3D 배열로 변환
# x_train = np.array(x_train)
# x_test = np.array(x_test)

# LSTM 모델 정의
model = Sequential()

model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(LSTM(64, activation='tanh', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh'))
model.add(Dense(6))
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae', 'mse', r2])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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