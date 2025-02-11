import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv('data/finedust_basic_data.csv')

# 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

def sequence_data(data, sequence_length=24):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(x), np.array(y)

# 데이터 분리
df_time = scaled_df[['year','month','day','hour','week']].values
df_pollutants = scaled_df[['no2', 'o3', 'co', 'so2', 'pm10', 'pm25']].values
df_weather = scaled_df[['wd', 'ws', 'ta', 'td', 'hm', 'rn', 'sd_tot', 'ca_tot', 'ca_mid', 'vs', 'ts', 'si', 'ps', 'pa']].values
    
x_pollutants, y_pollutants = sequence_data(df_pollutants, sequence_length=24)
x_weather, y_weather = sequence_data(df_weather, sequence_length=24)

x_train, x_test, y_train, y_test = train_test_split(x_pollutants, y_pollutants, test_size=0.3, shuffle=False)

model = Sequential()
# tanh (하이퍼볼릭 탄젠트) : 출력 범위가 -1과 1 사이, 음수 값을 포함하는 데이터 처리 시 유리함
# input_shape=(X_train.shape[1], X_train.shape[2]): 입력 데이터의 형태를 지정
# X_train.shape에서 1은 시퀀스 길이 (시간 단계), 2는 피처의 수(각 시점에서의 변수 수)
model.add(LSTM(64, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
    )

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# 모델 평가
test_loss = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')

# 결과 시각화
y_pred = model.predict(x_test)
plt.plot(y_test[:50], label='True values')
plt.plot(y_pred[:50], label='Predicted values')
plt.legend()
plt.show()

