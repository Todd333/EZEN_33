from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/calc')
def calculate():
    a = request.args.get('num1','0')

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
