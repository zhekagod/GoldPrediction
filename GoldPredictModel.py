# С:<
import clearml
from clearml import Task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from tsfresh import extract_features
from sklearn import model_selection
from sklearn import linear_model as lm
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def windows_extracting(ds, data_window_size=30, label_window_size=7):
    """Возвращает список всех промежутков по <data_window_size> строк,
       и список соответствующих последующих промежутков по <label_window_size> строк"""
    # Создание списков, содержащих промежутки данных по <data_window_size> дней
    data_windows = []
    # Всего должно быть len(dataset) - data_window_size + 1 - label_window_size списков; индексация от 0
    for i in range(len(ds) - data_window_size + 1 - label_window_size):
        data_windows.append(ds[i:i + data_window_size])  # exclusive!

    # Создание списков с метками (y) для каждого промежутка, содержащих <label_window_size> последующих дней
    label_windows = []
    for i in range(data_window_size, len(data_windows) + data_window_size):
        label_windows.append(ds[i:i + label_window_size]['<CLOSE>'])
    # Проверка, что длины data_windows и label_windows совпадают
    assert len(data_windows) == len(label_windows)

    # Проверка, что последние промежутки не "зажевались"
    assert len(data_windows[-1]) == data_window_size and len(label_windows[-1]) == label_window_size

    return data_windows, label_windows

def validate_result(model, model_name, validation_X, validation_y):
    # validation_X = validation_X[0:-1:7]
    # validation_y = validation_y[0:-1:7]
    plt.figure(figsize=(10, 6))
    predicted_data = []
    real_data = []
    losses = []
    for i in range(0, len(validation_X), 7):
        y_pred = model.predict([validation_X[i]])
        predicted_data.extend(y_pred[0])
        real_data.extend(validation_y[i])

        # Оценка модели
        # mse = mean_squared_error([y_test[i]], y_pred)
        mape = mean_absolute_percentage_error(validation_y[i], y_pred)
        losses.append(mape)
    # predicted = model.predict(validation_X)
    RSME_score = np.sqrt(mean_squared_error(real_data, predicted_data))
    print('RMSE: ', RSME_score)

    R2_score = r2_score(real_data, predicted_data)
    print('R2 score: ', R2_score)
    validation_y = validation_y[0:-1:7]
    indexes = np.concatenate([df.index for df in validation_y])
    # print(indexes)
    plt.plot(indexes, predicted_data, 'r', label='Предсказания')  # validation_y.index, 
    plt.plot(indexes, real_data, 'b', label='Реальные данные')
    # Установка значений для оси X
    plt.ylabel('Цена закрытия')
    # Установка шага для отображения дат
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.title(model_name + ' Предсказания и реальные дынные')
    plt.legend(loc='upper right')
    plt.show()


def main():
    # Task.init(project_name="GoldPrediction", task_name="1step")
    # Импорт датасетов. Формат даты в исходниках yymmdd преобразуется с помощью parse_dates
    df1 = pd.read_csv("GoldPrediction/DataSources/GC_070904_090320.csv", parse_dates=[0],
                      usecols=lambda x: x != '<TIME>', index_col=['<DATE>'])  # , index_col=['<DATE>']
    df2 = pd.read_csv("GoldPrediction/DataSources/GC_090321_140320.csv", parse_dates=[0],
                      usecols=lambda x: x != '<TIME>', index_col=['<DATE>'])
    df3 = pd.read_csv("GoldPrediction/DataSources/GC_140321_190320.csv", parse_dates=[0],
                      usecols=lambda x: x != '<TIME>', index_col=['<DATE>'])
    df4 = pd.read_csv("GoldPrediction/DataSources/GC_190321_240319.csv", parse_dates=[0],
                      usecols=lambda x: x != '<TIME>', index_col=['<DATE>'])
    print(df1.head())

    # Объединение датасетов
    dataset = pd.concat([df1, df2, df3, df4])  # , ignore_index=True
    # Удаление повторяющихся строк
    dataset = dataset[~dataset.index.duplicated()]
    # print(dataset.isnull().sum())
    scaler = MinMaxScaler(feature_range=(0, 1))
    """dataset[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']] = (
        scaler.fit_transform(dataset[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]))"""

    print(dataset)
    print(dataset.head())
    print("dataset.shape: ", dataset.shape)
    # X_train, Y_train, x_test, y_test = model_selection.train_test_split(dataset)
    """plt.plot(dataset['<CLOSE>'], label="Close values")
    plt.legend()
    plt.show()"""

    """features_data = dataset

    features_data['<PRICE_CHANGE>'] = dataset['<CLOSE>'] - dataset['<OPEN>']
    features_data['<PRICE_RANGE>'] = dataset['<HIGH>'] - dataset['<LOW>']"""
    
    
    features_ds = pd.read_csv("GoldPrediction\DataSources\extracted_features_cleaned.csv").iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 45, 55, 67, 113, 114]]
    features_ds.index = dataset.index
    print("features_ds.shape: ",features_ds.shape)
    print(features_ds.head())
    print(features_ds.iloc[1])
    print("=====features dataset concat=====")
    print(dataset.head())
    # dataset = dataset.join(features_ds)
    dataset = pd.concat([dataset, features_ds], axis=1)
    # dataset.to_csv('GoldPrediction\DataSources\concat_data.csv')
    # dataset = dataset.merge(dataset, left_index=True, right_index=True)
    # print(dataset.columns())
    print(dataset.iloc[1])
    print(dataset.head())

    # Получение списка всех колонок датасета
    columns = dataset.columns

    # Применение scaler к каждой колонке отдельно
    for column in columns:
        dataset[column] = scaler.fit_transform(dataset[[column]])
    # dataset.to_csv('GoldPrediction\DataSources\concat_data.csv')
    data_windows, label_windows = windows_extracting(dataset, data_window_size=30, label_window_size=7)

    """print("Пример последнего промежутка данных и соответствующих меток:")
    print(data_windows[-1])
    print(label_windows[-1])

    validation_X, validation_y = data_windows[-1], label_windows[-1]

    X_train, y_train = data_windows[:-1], label_windows[:-1]
    print("validation_X, validation_y: \n", validation_X, validation_y)
    print(len(validation_X), len(validation_y))
    print(len(dataset[0:1000]), len(dataset[1000:2000]))


    lasso_clf = LassoCV(n_alphas=1000, max_iter=10000, random_state=0)
    lasso_clf_feat = lasso_clf.fit(dataset[0:2500], dataset['<CLOSE>'][2500:5000])  # data_windows[:-1], label_windows[:-1]
    validate_result(lasso_clf_feat, 'LassoCV', validation_X, validation_y)"""
    # Преобразование списков окон данных и меток в массивы numpy
    X = data_windows
    y = label_windows

    # Получение размеров списков X и y
    num_windows = len(X)
    print("Количество окон в X:", num_windows)
    print("Количество окон в y:", len(y))

    # Разделение данных на обучающий и тестовый наборы
    split = num_windows // 2
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    # model = lm.MultiTaskLassoCV(n_alphas=1000, max_iter=10000, random_state=0)
    # model = lm.RidgeCV()
    # Преобразование X_train и X_test в двумерные списки

    # Строка: все признаки 1-го дня, все признаки 2-го дня, все признаки 3-го дня, ...
    X_train_flat = np.array([np.array(window.values.flatten()) for window in X_train])
    print("Строка датасета: ", X_train_flat[0])
    print("Длина строки тренировочного датасета: ", len(X_train_flat[0]))
    X_test_flat = np.array([np.array(window.values.flatten()) for window in X_test])
    print("Обучение модели...")
    fitted_model = model.fit(X_train_flat, y_train)
    selected_model = SelectFromModel(fitted_model)
    selected_model.fit(X_train_flat, y_train)
    sup = selected_model.get_support()
    sup_indexes = np.where(sup == True)[0].tolist()
    print(sup_indexes, type(sup_indexes), type(sup_indexes[0]))
    X_train_flat = X_train_flat[:, sup_indexes]
    X_test_flat = X_test_flat[:, sup_indexes]
    model = model.fit(X_train_flat, y_train)
    # zipped = zip(X_train_flat,sup)
    # print(*zipped)
    # Предсказание на тестовом наборе
    #y_pred = model.predict(X_test_flat)
    # Списки для хранения предсказанных и реальных данных
    predicted_data = []
    real_data = []

    # Построение графиков предсказанных и реальных данных
    left_border = 200
    right_border = 400
    losses = []
    for i in range(left_border, right_border, 7):
        y_pred = model.predict([X_test_flat[i]])
        predicted_data.extend(y_pred[0])
        real_data.extend(y_test[i])

        # Оценка модели
        # mse = mean_squared_error([y_test[i]], y_pred)
        mape = mean_absolute_percentage_error(y_test[i], y_pred)
        losses.append(mape)
        # print("Средняя относительная ошибка (MAPE):", mape)
        # print([X_test_flat[i]])
        # print([y_test[-left_border+i]])
    total_loss = np.mean(losses)
    # Построение графика предсказанных и реальных данных
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_data, label='Предсказанные данные', color='red')
    plt.plot(real_data, label='Реальные данные', color='blue')
    plt.title('Сравнение реальных и предсказанных данных')
    plt.xlabel(f'Средняя ошибка: {total_loss}')
    plt.ylabel('Цена закрытия')
    plt.legend()
    plt.grid(True)
    plt.show()
    validate_result(model, "Linear", X_test_flat, y_test)
    """plt.plot(dataset['<VOL>'], label="Volatility values")
    plt.legend()
    plt.show()"""

    """extracted_features = extract_features(dataset, column_id='<DATE>')

    # Вывод извлеченных признаков на экран
    print(extracted_features)
    extracted_features.to_csv('DataSources/extracted_features.csv', index=False)"""



if __name__ == '__main__':
    main()