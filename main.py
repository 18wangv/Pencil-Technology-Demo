from next_word_prediction import GPT2


import numpy as np
from flask import Flask, request, render_template, make_response, redirect, url_for

model = GPT2()
app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('input_form.html')


@app.route('/', methods=['GET', 'POST'])
def my_form_post():
    text = request.form['text']
    response = get_prediction(text)
    return render_template('input_form.html', entered_text=text, predicted_text=response)

@app.route('/', methods=['POST'])
def delete_images():
    if request.method == 'POST':
        return redirect(url_for('my_form'))

def get_prediction(data):
    prediction = model.predict_next(data,5)  # runs globally loaded model on the data
    return str(prediction[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)