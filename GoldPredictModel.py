# С:<
import clearml
from clearml import Task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from tsfresh import extract_features
from sklearn import model_selection
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression


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
        label_windows.append(ds[i:i + label_window_size]['<CLOSE>'].values)
    # Проверка, что длины data_windows и label_windows совпадают
    assert len(data_windows) == len(label_windows)

    # Проверка, что последние промежутки не "зажевались"
    assert len(data_windows[-1]) == data_window_size and len(label_windows[-1]) == label_window_size

    return np.array(data_windows), np.array(label_windows)

def validate_result(model, model_name, validation_X, validation_y):
    predicted = model.predict(validation_X)
    RSME_score = np.sqrt(mean_squared_error(validation_y, predicted))
    print('RMSE: ', RSME_score)

    R2_score = r2_score(validation_y, predicted)
    print('R2 score: ', R2_score)

    plt.plot(predicted, 'r', label='Predict')
    plt.plot(validation_y, 'b', label='Actual')
    plt.ylabel('Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.title(model_name + ' Predict vs Actual')
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
    # print(dataset.isnull().sum())
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']] = (
        scaler.fit_transform(dataset[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']]))

    print(dataset)
    print(dataset.head())
    print(dataset.shape)

    # X_train, Y_train, x_test, y_test = model_selection.train_test_split(dataset)
    plt.plot(dataset['<CLOSE>'], label="Close values")
    plt.legend()
    plt.show()

    """features_data = dataset

    features_data['<PRICE_CHANGE>'] = dataset['<CLOSE>'] - dataset['<OPEN>']
    features_data['<PRICE_RANGE>'] = dataset['<HIGH>'] - dataset['<LOW>']"""

    data_windows, label_windows = windows_extracting(dataset, data_window_size=50, label_window_size=50)

    print("Пример последнего промежутка данных и соответствующих меток:")
    print(data_windows[-1])
    print(label_windows[-1])

    validation_X, validation_y = data_windows[-1], label_windows[-1]

    X_train, y_train = data_windows[:-1], label_windows[:-1]
    print("validation_X, validation_y: \n", validation_X, validation_y)
    print(len(validation_X), len(validation_y))
    print(len(dataset[0:1000]), len(dataset[1000:2000]))


    lasso_clf = LassoCV(max_iter=6000, random_state=0)
    lasso_clf_feat = lasso_clf.fit(dataset[0:2500], dataset['<CLOSE>'][2500:5000])  # data_windows[:-1], label_windows[:-1]
    validate_result(lasso_clf_feat, 'LassoCV', validation_X, validation_y)
    """plt.plot(dataset['<VOL>'], label="Volatility values")
    plt.legend()
    plt.show()"""

    """extracted_features = extract_features(dataset, column_id='<DATE>')

    # Вывод извлеченных признаков на экран
    print(extracted_features)
    extracted_features.to_csv('DataSources/extracted_features.csv', index=False)"""



if __name__ == '__main__':
    main()
