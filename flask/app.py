from flask import Flask, request, render_template
import setuptools.dist
import tensorflow as tf
import pickle 
import os
import numpy as np

app = Flask(__name__)

app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')


@app.route('/')
def metod():
    return render_template('main.html')


def predict(data):
    model_path = os.path.join('models', 'model_1 (1).h5')
    model = tf.keras.models.load_model(model_path)
    pred = model.predict([data])
    return pred


@app.route('/input/', methods=['POST', 'GET'])
def prediction():
    message = ''
    if request.method == 'POST':
        param_list = ('Плотность', 'Количество отвердителя', 'Содержание эпоксидных групп',
                      'Температура вспышки', 'Модуль упругости при растяжении',
                      'Потребление смолы', 'Плотность нашивки')
        params = []
        for x in param_list:
            param = request.form.get(x)
            if param is None:
                param = "0"
            params.append(param)
        params = np.array([float(x.replace(',', '.')) for x in params])
        scaler_path = os.path.join('models', 'scaler')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        scaler_params = scaler.transform([params])
        result = predict(scaler_params)
        y = result.flatten()
        
        message = f'Соотношение матрица-наполнитель: {y[0]:.3f}'
    return render_template('input.html', message=message)


if __name__ == '__main__':
    app.run()
