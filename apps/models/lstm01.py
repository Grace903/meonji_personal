import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt

df = pd.read_csv('data/finedust_basic_data_12.csv')

#원핫인코딩, 평일 주말 구분

# 1. 평일/주말 구분, 컬럼 추가
df['weekend'] = df['week'].isin([5, 6]).astype(int)  # 주말
df['weekday'] = df['weekend'].isin([0,1,2,3,4]).astype(int) # 평일

# 2. 'week' 컬럼 원핫인코딩
week_encoded = pd.get_dummies(df['week'], prefix='week')

# 3. 원핫인코딩된 데이터와 주말/평일 변수 결합
df = pd.concat([df, week_encoded], axis=1)

# 계절 column 추가 (season: 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울)
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

# 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop(['week'], axis=1))  # 'week' 컬럼 제외하고 스케일링
scaled_df = pd.DataFrame(scaled_data, columns=df.columns.drop('week'))  # 'week' 제외하고 새로운 DataFrame 생성

# 데이터 분리

df_latlon = scaled_df[['lat','lon']].values

# x축 - 시간, 요일, 계절, 바람, 온도, 습도, 일광, 전운량량
df_time = df[['year', 'month', 'day', 'hour'] + list(week_encoded.columns) + ['season']].values
df_week = df[['weekend','weekday']].values
df_wind = df[['wd', 'ws']].values  # 풍향, 풍속
df_air = df[['ps','pa']].values
df_temp = df[['ta', 'td', 'ts']].values  # 기온, 이슬점 온도, 지면 온도 
df_water = df[['rn', 'hm', 'sd_tot']].values  # 강수량, 상대습도 , 적설 
df_si = df['si'].values  # 일조량
df_cloud = df['ca_tot','ca_mid'].values # 전운량

# df_climate = scaled_df[['wd','ws','ta','td','hm','rn','sd_tot','ca_tot','ca_mid','ts','si','ps','pa']].values

# y축 - 미세먼지 농도와 성분인 'pm10', 'pm25', 'no2', 'o3', 'co', 'so2'
df_pm = scaled_df[['no2','o3','co','so2','pm10','pm25']].values




sequence_length = 24

def create_sequences(data, sequence_length=sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])  # 다음 시점의 값
    return np.array(x), np.array(y)

# 데이터를 시퀀스로 변환
x_time, y_time = create_sequences(df_time)
x_week, y_week = create_sequences(df_week)
x_wind, y_wind = create_sequences(df_wind)
x_air, y_air = create_sequences(df_air)
x_temp, y_temp = create_sequences(df_temp)
x_water, y_water = create_sequences(df_water)
x_si, y_si = create_sequences(df_si)
x_cloud, y_cloud = create_sequences(df_cloud)

# 미세먼지 농도 관련 데이터 시퀀스화
x_pm, y_pm = create_sequences(df_pm)

# 최종 데이터 합치기 (시계열 특성들을 하나로 합칩니다)
x = np.concatenate([x_time, x_week, x_wind, x_air, x_temp, x_water, x_si, x_cloud], axis=2)
y = y_pm  # y는 미세먼지 농도 (pm10, pm25, no2, o3, co, so2)

# 데이터 형태 확인
print(x.shape, y.shape)




columns_order = ['year','month', 'day', 'hour','week','weekend','weekday','season',
                'wd','ws','ta','td','hm','rn','sd_tot','ca_tot','ca_mid','vs','ts','si','ps','pa',
                'no2','o3','co','so2','pm10','pm25',]

df = df[columns_order]




#여기부터
#------------------------------------------------------------------------------------------------------------------------

def sequence_data(data, sequence_length=24): # 1주 = 168시간
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(x), np.array(y)

# R² 메트릭 정의
def r2(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

# Pollutants 데이터 시퀀스 생성
x_pollutants, y_pollutants = sequence_data(df_pm, sequence_length=24)

# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x_pollutants, y_pollutants, test_size=0.1, shuffle=False)

# LSTM 모델 정의
model = Sequential()
# tanh (하이퍼볼릭 탄젠트) : 출력 범위가 -1과 1 사이, 음수 값을 포함하는 데이터 처리 시 유리함
# input_shape=(X_train.shape[1], X_train.shape[2]): 입력 데이터의 형태를 지정
# X_train.shape에서 1은 시퀀스 길이 (시간 단계), 2는 피처의 수(각 시점에서의 변수 수)
model.add(LSTM(64, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh'))
model.add(Dense(6))

model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae','mse','mape','accuracy', r2])


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
plt.plot(y_pred[:50, 0], label='Predicted PM10')  # 예측 PM10 값
plt.legend()
plt.title('True vs Predicted PM10')

plt.subplot(1, 2, 2)
plt.plot(y_test[:50, 1], label='True PM25')  # 실제 PM25 값
plt.plot(y_pred[:50, 1], label='Predicted PM25')  # 예측 PM25 값
plt.legend()
plt.title('True vs Predicted PM25')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.title('Loss, MAE, MSE')
plt.legend()

plt.show()

# 모델 저장
model.save('finedust_lstm_test01.h5')