import flask
from flask import render_template
import numpy as np
import pickle


app = flask.Flask(__name__, template_folder = 'templates', static_folder='static')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('model/model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        print(flask.request.form)
        data = list(flask.request.form.values())

        y_pred = loaded_model.predict(np.array(data).reshape(1, -1).astype(float))

        return render_template('main.html', result = y_pred)

if __name__ == '__main__':
    app.run()

