from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
from app import main
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            days = int(request.form.get('prediction_days', 10))
            chart_path, forecast_df = main.run_forecast(days)

            # Perform EDA
            print("\n===== EDA Summary =====")
            df = pd.read_csv('data/deutsche_bank_financial_performance.csv')
            print("📊 Dataset Shape:", df.shape)
            print("\n🔍 Missing Values:\n", df.isnull().sum())
            print("\n📈 Summary Statistics:\n", df.describe())

            return redirect(url_for('results', chart_path=chart_path))

        except Exception as e:
            print("Error during forecast:", str(e))
            return render_template('index.html', error=str(e))

    return render_template('index.html')

@app.route('/results')
def results():
    chart_path = request.args.get('chart_path')
    return render_template('results.html', chart=chart_path)

@app.route('/output/<filename>')
def send_output_file(filename):
    return send_from_directory('output', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
