from flask import Flask, render_template, url_for, redirect, request
import pickle

app = Flask(__name__)
model = pickle.load(open('regressor_model.pkl', 'rb'))

@app.route('/')
def home():
	return redirect(url_for('predict_weight'))

@app.route('/predict_weight', methods = ['GET', 'POST'])
def predict_weight():
	if request.method == 'POST':
		height = request.form['height']
		height = float(height) / 2.54
		weight = model.predict([[height]]) 
		return render_template("index.html", weight = weight[0][0] / 2.205, unit = "kgs")
	return render_template("index.html")


if __name__ == '__main__':
	app.run(debug = True)