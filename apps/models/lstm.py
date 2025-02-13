import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv('data/finedust_basic_data_drop.csv')

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

x_train, x_test, y_train, y_test = train_test_split(x_pollutants, y_pollutants, test_size=0.2, shuffle=False)

# 모델 구성
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mean_squared_error')

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

# 손실 시각화
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# 모델 평가
test_loss = model.evaluate(x_test, y_test)
print(f'Test Loss (MSE): {test_loss}')

# 예측값 계산
y_pred = model.predict(x_test)

# 회귀 평가 지표 (MSE, MAE, R^2)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 평가 지표 출력
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R^2): {r2:.4f}')

# 결과 시각화
plt.plot(y_test[:50], label='True values')
plt.plot(y_pred[:50], label='Predicted values')
plt.legend()
plt.show()
