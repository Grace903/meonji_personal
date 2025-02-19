import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.layers import Dropout, LSTM, Dense, BatchNormalization, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint

import pickle


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
    
    return df, season_encoded


def dayOfTheWeek_col(df):
    df['weekend'] = df['week'].isin([5, 6]).astype(int)  # 주말
    df['weekday'] = df['week'].isin([0, 1, 2, 3, 4]).astype(int)  # 평일

    # 요일 원핫인코딩
    week_encoded = pd.get_dummies(df['week'], prefix='week', drop_first=False)  # 요일 원핫인코딩
    df = pd.concat([df, week_encoded], axis=1)
    
    return df, week_encoded


# 계절 및 요일 컬럼 추가
df, season_encoded = season_col(df)
df, week_encoded = dayOfTheWeek_col(df)

df['season_week_interaction'] = df['season'] * df['weekend']


# 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)


# 원인 데이터
def input_data(scaled_df, week_encoded, season_encoded):
    df_latlon = scaled_df[['lat', 'lon']]
    df_time = scaled_df[['year', 'month', 'day', 'hour'] + list(week_encoded.columns) + list(season_encoded.columns)]
    df_climate = scaled_df[['wd', 'ws', 'ta', 'td', 'hm', 'rn', 'sd_tot', 'ca_tot', 'ca_mid', 'vs', 'ts', 'si', 'ps', 'pa']]

    df_interaction = scaled_df[['season_week_interaction']]  # 이 부분 추가

    
    # x는 입력 변수들, y는 결과 변수
    x = pd.concat([df_latlon, df_time, df_climate, df_interaction], axis=1)  # x에는 lat, lon, 시간, 날씨 등의 정보
    y = scaled_df[['pm10', 'pm25', 'no2', 'o3', 'co', 'so2']]  # y는 미세먼지 농도 (결과)
    
    return x, y


# 데이터 분리
x, y = input_data(scaled_df, week_encoded, season_encoded)


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
        
        if end > len(x):
            break
        
        x_test.append(x.iloc[start:end, :].values)  # (window_size, features)
        y_test.append(y.iloc[start:end, :].values)  # (window_size, targets)

    # 리스트를 numpy 배열로 변환
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


# 슬라이딩 윈도우 분할
window_size = 24
step_size = 1

# x.values와 y.values를 사용하여 슬라이딩 윈도우로 분할
x_train, y_train, x_test, y_test = sliding_window_split(x, y, window_size, step_size)


# LearningRateScheduler 콜백 정의
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.8
    else:
        return lr * 0.5

lr_scheduler = LearningRateScheduler(scheduler)



# LSTM 모델 정의
model = Sequential()
model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l2(0.01))))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(LSTM(32, activation='tanh', return_sequences=True))
model.add(Dense(32, kernel_initializer='glorot_normal'))
model.add(Dense(6))
model.add(BatchNormalization())
model.add(Dropout(0.3))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)


# 모델 훈련
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, lr_scheduler, reduce_lr, model_checkpoint]
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

# 예측값을 2D로 변환 (LSTM의 출력은 3D 형태이므로)
y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])



# 모델 저장 경로 (예시: 현재 작업 디렉토리에 저장)
model_path = 'pm_lstm_sw04.keras'

# 모델 저장
model.save(model_path)

# 모델 저장
# model.save('pm_lstm_model.keras')

# 전처리 저장 (StandardScaler)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 컬럼 순서 저장
with open('columns.pkl', 'wb') as f:
    pickle.dump(x.columns, f)

# 원핫 인코딩된 데이터 컬럼 저장
with open('season_encoded.pkl', 'wb') as f:
    pickle.dump(season_encoded, f)

with open('week_encoded.pkl', 'wb') as f:
    pickle.dump(week_encoded, f)
