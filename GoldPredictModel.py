# С:<
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
# from tsfresh import extract_features
from sklearn import model_selection
from sklearn import linear_model as lm
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        selected_features = []
        for window_array in X:
            selected_window_features = window_array[self.feature_indices]
            selected_features.append(np.array(selected_window_features))
        return np.array(selected_features)


class FlattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_flat = np.array([np.array(window.values.flatten()) for window in X])
        return X_flat


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


def validate_result(model, model_name, validation_X, validation_y, pred_window_size):
    # validation_X = validation_X[0:-1:pred_window_size]
    # validation_y = validation_y[0:-1:pred_window_size]
    plt.figure(figsize=(10, 6))
    predicted_data = []
    real_data = []
    losses = []
    validation_X = validation_X[0:-1:pred_window_size]
    validation_y = validation_y[0:-1:pred_window_size]
    for i in range(0, len(validation_X)):
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
    indexes = np.concatenate([df.index for df in validation_y])
    # print(indexes)
    plt.plot(indexes, predicted_data, 'r', label='Предсказания')  # validation_y.index,
    plt.plot(indexes, real_data, 'b', label='Реальные данные')
    # Установка значений для оси X
    plt.ylabel('Цена закрытия')
    # Установка шага для отображения дат
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.title(model_name + ' Предсказания и реальные данные')
    plt.legend(loc='upper right')
    plt.show()
    plt.plot(losses)
    plt.xlabel(f'Средняя ошибка (%): {np.mean(losses)}')
    plt.show()


def add_features(dataset):

    features_ds = pd.read_csv("GoldPrediction/DataSources/extracted_features_cleaned.csv").iloc[:,
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
    features_ds.index = dataset.index
    print("features_ds.shape: ", features_ds.shape)
    print(features_ds.head())
    print(features_ds.iloc[1])
    print("=====features dataset concat=====")
    # dataset = dataset.join(features_ds)
    dataset = pd.concat([dataset, features_ds], axis=1)

    dataset['<PRICE_CHANGE>'] = dataset['<CLOSE>'] - dataset['<OPEN>']
    dataset['<PRICE_RANGE>'] = dataset['<HIGH>'] - dataset['<LOW>']

    # Скользящее среднее за последние <window> дней.
    dataset['<MA_7>'] = dataset['<CLOSE>'].rolling(window=7).mean()

    # Экспоненциальное скользящее среднее за последние <span> дней.
    dataset['<EMA_14>'] = dataset['<CLOSE>'].ewm(span=14, min_periods=0, adjust=False).mean()

    # Скользящее стандартное отклонение за последние <window> дней.
    dataset['<STD_30>'] = dataset['<CLOSE>'].rolling(window=30).std()

    # Лаги цен за последние <i in range(1, 4)> дней.
    for i in range(1, 4):
        dataset[f'<Price_Lag_{i}>'] = dataset['<CLOSE>'].shift(i)

    dataset.fillna(0, inplace=True)
    return dataset


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

    """plt.plot(dataset['<CLOSE>'], label="Close values")
    plt.legend()
    plt.show()"""

    dataset = add_features(dataset)

    # dataset.to_csv('GoldPrediction\DataSources\concat_data.csv')
    # dataset = dataset.merge(dataset, left_index=True, right_index=True)
    # print(dataset.columns())

    print("features added")
    # Печать первых нескольких строк для проверки.
    print(dataset.head())
    print(dataset.iloc[1])
    print(dataset.head())

    # Получение списка всех колонок датасета
    columns = dataset.columns

    # Применение scaler к каждой колонке отдельно
    for column in columns:
        dataset[column] = scaler.fit_transform(dataset[[column]])
    # dataset.to_csv('GoldPrediction\DataSources\concat_data.csv')

    # Определили оценками маловажные признаки и теперь их убираем
    # dataset.drop(dataset.columns[[5, 6, 9]], axis=1, inplace=True)
    # !!! сколько дней нужно предсказать
    pred_window_size = 1
    data_windows, label_windows = windows_extracting(dataset, data_window_size=30, label_window_size=pred_window_size)

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
    # model = LinearRegression()
    # model = lm.MultiTaskLassoCV(n_alphas=1000, max_iter=6000, random_state=0)
    # model = lm.RidgeCV()
    # Преобразование X_train и X_test в двумерные списки

    # Конвейер, чтобы преобразовать данные в одномерные массивы внутри модели, а снаружи передавать массив.
    steps = [
        ('flat', FlattenTransformer()),  # Преобразование в одномерные массивы
        # ('feature_extractor', FeatureExtractor(np.arange(390))),  # Извлечение нужных признаков
        # ('feature_selector', SelectKBest(score_func=f_regression, k=300)),
        ('feature_selection', SelectFromModel(LinearRegression())),
        # ('model', LinearRegression())  # Модель
        ('model', lm.MultiTaskLassoCV(n_alphas=1000, max_iter=6000, random_state=0))  # Модель
    ]

    # Создаем конвейер, передавая список этапов.
    model = Pipeline(steps)
    # Строка: все признаки 1-го дня, все признаки 2-го дня, все признаки 3-го дня, ...
    X_train_flat = np.array([np.array(window.values.flatten()) for window in X_train])
    print("Строка датасета: ", X_train_flat[0])
    print("Длина строки тренировочного датасета: ", len(X_train_flat[0]))
    X_test_flat = np.array([np.array(window.values.flatten()) for window in X_test])

    print("Обучение модели...")
    # fitted_model = model.fit(X_train, y_train)

    # selected_model = SelectFromModel(fitted_model, prefit=True)
    # selected_model.fit(X_train, y_train)
    """sup = model['feature_selector'].get_support()
    sup_indexes = np.where(sup == True)[0].tolist()
    print(sup_indexes, type(sup_indexes), type(sup_indexes[0]))
    marks = np.zeros(13)
    for i in sup_indexes:
        marks[i % 13] += 1
    print(marks)"""
    # X_train_flat = X_train_flat[:, sup_indexes]
    # X_test_flat = X_test_flat[:, sup_indexes]
    model = model.fit(X_train, y_train)
    # feature_importances = np.where(model.named_steps['feature_selector'].get_support()==True)[0].tolist()
    """for feature, importance in enumerate(feature_importances):
        print(f"Feature {feature}: Importance = {importance}")"""
    # Списки для хранения предсказанных и реальных данных
    predicted_data = []
    real_data = []

    # Построение графиков предсказанных и реальных данных
    left_border = 200
    right_border = 400
    losses = []
    print("Тест модели")
    for i in range(left_border, right_border, pred_window_size):
        y_pred = model.predict([X_test[i]])
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
    plt.xlabel(f'Средняя ошибка  (%): {total_loss}')
    plt.ylabel('Цена закрытия')
    plt.legend()
    plt.grid(True)
    plt.show()
    validate_result(model, "Linear", X_test, y_test, pred_window_size)
    """plt.plot(dataset['<VOL>'], label="Volatility values")
    plt.legend()
    plt.show()"""

    """extracted_features = extract_features(dataset, column_id='<DATE>')

    # Вывод извлеченных признаков на экран
    print(extracted_features)
    extracted_features.to_csv('DataSources/extracted_features.csv', index=False)"""

def features_work(dataset):
    indexed_dataset = dataset.copy()
    """def row_index_comeback(row):
        return row.index"""
    indexed_dataset = indexed_dataset.assign(row_number=range(len(indexed_dataset)))
    print('============================')
    print(indexed_dataset)
    print('============================')
    # ef = extract_features(indexed_dataset, column_id='row_number')
    ef = pd.read_csv("DataSources/extracted_features.csv")
    # Удаление столбцов, состоящих только из нулей
    # ef = ef.loc[:, (ef != 0).any(axis=0)]
    # Удаление столбцов, в которых Nan превышает 75%, оставшиеся Nan заполняет значениями 0
    ef = ef.dropna(axis=1, thresh=int(len(ef)/4)-1).fillna(0)  # how="all",
    # Удаление столбцов с одинаковыми значениями во всех строкаж
    ef = ef[[i for i in ef if ef[i].nunique() > 1]]
    """uniq_counts = ef.value_counts()
    print(uniq_counts)"""
    # Удаление столбцов, в 75% строках которых стоит значение 0 или 1
    print(len(ef.columns))
    ef = ef[ef.columns[(ef == 0).mean() < 0.75]]
    print(len(ef.columns))
    ef = ef[ef.columns[(ef == 1).mean() < 0.75]]
    print(len(ef.columns))
    """nunique = ef.nunique()
    cols_to_drop = nunique[nunique == 1].index
    ef = ef.drop(cols_to_drop, axis=1)"""

    print(ef.head())
    print(ef.columns.tolist())
    print(ef.shape)
    # ef.to_csv('DataSources/extracted_features_cleaned.csv', index=False)

if __name__ == '__main__':
    main()