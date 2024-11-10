Прогнозирование конечных свойств новых композиционных материалов

МГТУ им. Н.Э. Баумана

Целью данной работы является разработка пользовательского приложения для прогнозирования ряда  конечных  свойств  композиционных материалов.

- Проведен разведочный анализ и представлена визуализация предложенных данных. Представлены гистограммы распределения каждой из переменной, диаграммы ящика с усами. 

- Проведена предобработка данных (удалены выбросы, нормализация и т.д.).

- Обучено нескольких моделей для прогноза модуля упругости при растяжении и прочности при растяжении. При построении модели было 20% данных оставлено на тестирование модели, на остальных происходило обучение моделей:

1 Lasso	
2	Ridge	
3	HuberRegressor	
4	LinearRegression	
5	GradientBoostingRegressor	
6	RandomForestRegressor	
7	XGBRegressor	
8	KNeighborsRegressor	
9	ExtraTreeRegressor	
10	DecisionTreeRegressor

- В качестве метрик RMSE, MAE, MAPE, R2

- Разработана нейронная сеть, которая будет рекомендовать "Соотношение матрица-наполнитель".

- Разработано пользовательское приложение на Flask, рекомендуемое "Соотношение матрица - наполнитель".

- Оценена точность модели на тренировочном и тестовом датасете.

- Создан репозиторий в GitHub и размещен код исследования.



Инструкция использования приложения:

Приложение позволяет решать задачу прогнозирования "Соотношение матрица - наполнитель". Для получения прогноза необходимо

 • запустить app.py,

 • в появившейся строке ( * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)) - нажать на ссылку: http://127.0.0.1:5000/.

 • в новом окне нажать "прогнозирование", далее ввести параметры и "рассчитать".



Выпускная квалификационная работа по программе повышения квалификации «Data Science» в обучающем центре МГТУ им. Н. Э. Баумана 2024 г.

