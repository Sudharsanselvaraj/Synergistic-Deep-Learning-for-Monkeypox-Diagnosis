from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Main homepage

@app.route('/chatinterface')
def chatinterface():
    return render_template('chatinterface.py')  # The page you want to show

if __name__ == '__main__':
    app.run(debug=True)

