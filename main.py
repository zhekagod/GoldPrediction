import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates


def calculate_MACD(df, nslow=26, nfast=12):
    emaslow = df.ewm(span=nslow, min_periods=nslow, adjust=True, ignore_na=False).mean()
    emafast = df.ewm(span=nfast, min_periods=nfast, adjust=True, ignore_na=False).mean()
    dif = emafast - emaslow
    MACD = dif.ewm(span=9, min_periods=9, adjust=True, ignore_na=False).mean()
    return dif, MACD


def calculate_RSI(df, periods=14):
    # wilder's RSI
    delta = df.diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    rUp = up.ewm(com=periods,adjust=False).mean()
    rDown = down.ewm(com=periods, adjust=False).mean().abs()

    rsi = 100 - 100 / (1 + rUp / rDown)
    return rsi


def calculate_SMA(df, peroids=15):
    SMA = df.rolling(window=peroids, min_periods=peroids, center=False).mean()
    return SMA


def calculate_BB(df, peroids=15):
    STD = df.rolling(window=peroids,min_periods=peroids, center=False).std()
    SMA = calculate_SMA(df)
    upper_band = SMA + (2 * STD)
    lower_band = SMA - (2 * STD)
    return upper_band, lower_band


def calculate_stdev(df, periods=5):
    STDEV = df.rolling(periods).std()
    return STDEV


def validate_result(model, model_name):
    predicted = model.predict(validation_X)
    RSME_score = np.sqrt(mean_squared_error(validation_y, predicted))
    print('RMSE: ', RSME_score)

    R2_score = r2_score(validation_y, predicted)
    print('R2 score: ', R2_score)

    plt.plot(validation_y.index, predicted, 'r', label='Predict')
    plt.plot(validation_y.index, validation_y, 'b', label='Actual')
    plt.ylabel('Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.title(model_name + ' Predict vs Actual')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    df_final = pd.read_csv("DataSources\FINAL_USO.csv", index_col=['Date'])  # , parse_dates=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    print(df_final.head())
    """plt.plot(df_final['Close'], label="Close values")
    plt.legend()
    plt.show()"""
    GLD_adj_close = df_final['Adj Close']
    SPY_adj_close = df_final['SP_Ajclose']
    DJ_adj_close = df_final['DJ_Ajclose']

    """df_p = pd.DataFrame({'GLD': GLD_adj_close, 'SPY': SPY_adj_close, 'DJ': DJ_adj_close})
    df_ax = df_p.plot(title='Effect of Index prices on gold rates', figsize=(15, 8))

    df_ax.set_ylabel('Price')
    df_ax.legend(loc='upper left')
    plt.show()"""
    test = df_final

    SMA_GLD = calculate_SMA(GLD_adj_close)

    # Calculate Bollinger Bands for GLD
    upper_band, lower_band = calculate_BB(GLD_adj_close)

    # Calculate MACD for GLD
    DIF, MACD = calculate_MACD(GLD_adj_close)

    # Calculate RSI for GLD
    RSI = calculate_RSI(GLD_adj_close)

    # Calculating Standard deviation for GLD
    STDEV = calculate_stdev(GLD_adj_close)

    Open_Close = df_final.Open - df_final.Close

    High_Low = df_final.High - df_final.Low

    test['SMA'] = SMA_GLD
    test['Upper_band'] = upper_band
    test['Lower_band'] = lower_band
    test['DIF'] = DIF
    test['MACD'] = MACD
    test['RSI'] = RSI
    test['STDEV'] = STDEV
    test['Open_Close'] = Open_Close
    test['High_Low'] = High_Low

    # Dropping first 33 records from the data as it has null values because of introduction of technical indicators
    test = test[33:]

    # Target column
    target_adj_close = pd.DataFrame(test['Adj Close'])

    # selecting Feature Columns
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'SP_open', 'SP_high', 'SP_low', 'SP_Ajclose', 'SP_volume',
                       'DJ_open', 'DJ_high', 'DJ_low', 'DJ_Ajclose', 'DJ_volume', 'EG_open', 'EG_high', 'EG_low',
                       'EG_Ajclose', 'EG_volume', 'EU_Price', 'EU_open', 'EU_high', 'EU_low', 'EU_Trend', 'OF_Price',
                       'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend', 'OS_Price', 'OS_Open', 'OS_High',
                       'OS_Low', 'OS_Trend', 'SF_Price', 'SF_Open', 'SF_High',
                       'SF_Low', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Open', 'USB_High', 'USB_Low', 'USB_Trend',
                       'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low',
                       'PLT_Trend', 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low', 'PLD_Trend', 'RHO_PRICE',
                       'USDI_Price', 'USDI_Open', 'USDI_High',
                       'USDI_Low', 'USDI_Volume', 'USDI_Trend', 'GDX_Open', 'GDX_High',
                       'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume', 'USO_Open',
                       'USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume', 'SMA', 'Upper_band',
                       'Lower_band', 'DIF', 'MACD', 'RSI', 'STDEV', 'Open_Close', 'High_Low']

    feature_minmax_transform_data = scaler.fit_transform(test[feature_columns])
    feature_minmax_transform = pd.DataFrame(columns=feature_columns, data=feature_minmax_transform_data,
                                            index=test.index)
    print(feature_minmax_transform.head())

    # Shift target array because we want to predict the n + 1 day value

    target_adj_close = target_adj_close.shift(-1)
    validation_y = target_adj_close[-90:-1]
    target_adj_close = target_adj_close[:-90]

    # Taking last 90 rows of data to be validation set
    validation_X = feature_minmax_transform[-90:-1]
    feature_minmax_transform = feature_minmax_transform[:-90]

    ts_split = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in ts_split.split(feature_minmax_transform):
        X_train, X_test = feature_minmax_transform[:len(train_index)], feature_minmax_transform[len(train_index): (
                    len(train_index) + len(test_index))]
        y_train, y_test = target_adj_close[:len(train_index)].values.ravel(), target_adj_close[len(train_index): (
                    len(train_index) + len(test_index))].values.ravel()

    lasso_clf = LassoCV(n_alphas=1000, max_iter=4000, random_state=0)
    print(len(X_train), len(y_train))
    lasso_clf_feat = lasso_clf.fit(X_train, y_train)
    validate_result(lasso_clf_feat, 'LassoCV')