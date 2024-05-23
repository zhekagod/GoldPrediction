# GoldPrediction

Проект по прогнозированию цены на золото с использованием Python и различных сторонних библиотек, таких как numpy, pandas, matplotlib, statsmodels, sklearn, sktime и pytorch. Данные отслеживаются и отчеты создаются в ClearML.

# Описание проекта
Целью данного проекта является разработка модели прогнозирования цены на золото на основе исторических данных. Мы используем различные методы анализа данных и машинного обучения для предсказания будущих цен на золото.

# Состав команды
[Developer of forecasts and statistics](https://github.com/zhekagod)

[Frontend developer](https://github.com/Deolys)

[Developer of forecasts](https://github.com/Amaimonn)

[Visual and statistic analyzer](https://github.com/lowskillmaster)

[Visual and statistic analyzer](https://github.com/jamorAA)

# Установка и запуск проекта

Склонируйте репозиторий:
```bash
git clone https://github.com/zhekagod/GoldPrediction/...git
```

Установите зависимости:
```bash
pip install -r requirements.txt
```

Запустите скрипт для анализа данных и построения моделей прогнозирования:
```bash
python main.py
```
# Структура проекта
main: Реализация последней рабочей версии прогнозирующей программы с моделью LSTM

front: Реализация приложения на Streamlit

step2: Реализация модели MultiTaskLassoCV

step3: Реализация модели Sklearn Pipeline

step4: Реализация модели SARIMAX.

# Отчеты в ClearML
Отчеты о прогнозировании цены на золото параллельно создаются в ClearML. Вы можете найти подробные результаты экспериментов, включая метрики моделей и графики, [здесь](https://app.clear.ml/projects/f3cfb7e55b9b496c9f9c8747eec0be76/experiments/ffed92fa668e4b4b899dae5a681a1eb7/execution?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=)



