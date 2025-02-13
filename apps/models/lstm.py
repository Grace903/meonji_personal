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

week_encoded = pd.get_dummies(df['week'], prefix='week')  # 요일 원핫인코딩
df = pd.concat([df, week_encoded], axis=1)

# 주말 여부 추가
df['weekend'] = (df['week'] == 5) | (df['week'] == 6)  

# 계절 정보 추가 (season: 1 = 봄, 2 = 여름, 3 = 가을, 4 = 겨울)
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


def sequence_data(data, sequence_length=168): # 1주 = 168시간
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

# 데이터 분리
df_latlon = scaled_df[['lat','lon']].values
df_time = scaled_df[['year','month','day','hour','season','weekend']].values  # 계절과 주말 정보 추가
df_climate = scaled_df[['wd','ws','ta','td','hm','rn','sd_tot','ca_tot','ca_mid','vs','ts','si','ps','pa']].values
df_pm = scaled_df[['pm10', 'pm25', 'no2', 'o3', 'co', 'so2']].values

# Pollutants 데이터 시퀀스 생성
x_pollutants, y_pollutants = sequence_data(df_pm, sequence_length=168)

# 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x_pollutants, y_pollutants, test_size=0.3, shuffle=False)

# LSTM 모델 정의
model = Sequential()
# tanh (하이퍼볼릭 탄젠트) : 출력 범위가 -1과 1 사이, 음수 값을 포함하는 데이터 처리 시 유리함
# input_shape=(X_train.shape[1], X_train.shape[2]): 입력 데이터의 형태를 지정
# X_train.shape에서 1은 시퀀스 길이 (시간 단계), 2는 피처의 수(각 시점에서의 변수 수)
model.add(LSTM(128, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
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
plt.plot(y_pred[:50, 0], label='Predicted PM10')  # 예측 PM10 값
plt.legend()
plt.title('True vs Predicted PM10')

plt.subplot(1, 2, 2)
plt.plot(y_test[:50, 1], label='True PM25')  # 실제 PM25 값
plt.plot(y_pred[:50, 1], label='Predicted PM25')  # 예측 PM25 값
plt.legend()
plt.title('True vs Predicted PM25')

plt.show()

# 모델 저장
model.save('finedust_lstm_test03.h5')