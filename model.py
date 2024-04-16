import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf

# Загрузка данных о ценах на золото
data = pd.read_csv('datas\GC_200101_240410.csv', delimiter=';')

# Преобразование данных в формат временного ряда
data['Datetime'] = pd.to_datetime(data['<DATE>'].astype(str), format='%y%m%d')
data.set_index('Datetime', inplace=True)
data = data[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]

##########
def add_features(data):
    # 1. Технические индикаторы: скользящие средние за 7 и 30 дней
    data['SMA7'] = data['<CLOSE>'].rolling(window=7).mean()
    data['SMA30'] = data['<CLOSE>'].rolling(window=30).mean()
    
    # 2. Индекс относительной силы (RSI)
    delta = data['<CLOSE>'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))
    
    # 3. MACD (Moving Average Convergence Divergence) технический индикатор, 
    # позволяющий оценивать силу тренда и построенный с учетом усредненного 
    # изменения цены.
    exp1 = data['<CLOSE>'].ewm(span=12, adjust=False).mean()
    exp2 = data['<CLOSE>'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    
    # 4. Объем торгов
    data['Volume'] = data['<VOL>']
    
    # Удаление строк с пропущенными значениями
    data.dropna(inplace=True)
    
    return data

data = add_features(data)
##########

# Разделение данных на обучающий и тестовый наборы
train_size = int(len(data) * 0.8)
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# # Создание временных последовательностей для обучения и тестирования
# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:(i + seq_length)])
#         y.append(data[i + seq_length])
#     return np.array(X), np.array(y)

# seq_length = 20
# X_train, y_train = create_sequences(train_data_scaled, seq_length)
# X_test, y_test = create_sequences(test_data_scaled, seq_length)

# # Создание и обучение модели LSTM
# model = Sequential([
#     LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])),
#     Dropout(0.2),
#     Dense(10)  # соответствует количеству признаков (OHLCV)
# ])

# model.compile(optimizer='adam', loss='mse')
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# # Прогнозирование на тестовом наборе
# y_pred = model.predict(X_test)

# # Инвертирование нормализации
# y_test_inv = scaler.inverse_transform(y_test)
# y_pred_inv = scaler.inverse_transform(y_pred)

# # Расчет метрик
# mse = mean_squared_error(y_test_inv, y_pred_inv)
# mae = mean_absolute_error(y_test_inv, y_pred_inv)
# r2 = r2_score(y_test_inv, y_pred_inv)
# print('Mean Squared Error:', mse)
# print('Mean Absolute Error:', mae)
# print('R^2 Score:', r2)

# # Визуализация фактических и предсказанных значений
# plt.figure(figsize=(10, 6))
# plt.plot(test_data.index[seq_length:], y_test_inv[:, 3], label='Actual Prices')
# plt.plot(test_data.index[seq_length:], y_pred_inv[:, 3], label='Predicted Prices')  # Изменено на [:, 3] для получения закрытия (CLOSE)
# plt.xlabel('Date')
# plt.ylabel('Gold Prices')
# plt.title('Actual vs Predicted Gold Prices')
# plt.legend()
# plt.show()

# Создание временных последовательностей для обучения и тестирования
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X_train, y_train = create_sequences(train_data_scaled[:-1], seq_length)
X_test, y_test = create_sequences(test_data_scaled[:-1], seq_length)

# Создание и обучение модели LSTM
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(10)  # Увеличим количество признаков до 10
])

def custom_loss(y_true, y_pred):
    # Штраф за предсказание, совпадающее с предыдущими ценами
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    shift_penalty = tf.reduce_mean(tf.square(y_pred[:, :-1] - y_true[:, 1:]))
    return mse_loss + shift_penalty

# Компиляция модели
model.compile(loss="mse",
              # optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              optimizer="SGD",
              metrics=['mse']
)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test)

# Инвертирование нормализации
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Расчет метрик
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R^2 Score:', r2)

# Визуализация фактических и предсказанных значений
plt.figure(figsize=(10, 6))
plt.plot(test_data.index[seq_length:-1], y_test_inv[:, 3], label='Actual Prices')
plt.plot(test_data.index[seq_length:-1], y_pred_inv[:, 3], label='Predicted Prices')  # Изменено на [:, 3] для получения закрытия (CLOSE)
plt.xlabel('Date')
plt.ylabel('Gold Prices')
plt.title('Actual vs Predicted Gold Prices')
plt.legend()
plt.show()