from flask import Flask, render_template, request, flash, redirect, session, make_response
from werkzeug.utils import secure_filename
import os
import time
import json

app = Flask(__name__)
app.secret_key = '@#$123456&*()'

ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static/csv')

# buat direktori untuk menyimpan gambar yang diupload
if not os.path.isdir(app.config['UPLOAD_FOLDER']): 
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ==================== Helper functions ==========================

# return secure filename yang telah di-append milisecond di depan filename
def get_secure_filename(filename):
    now = str(round(time.time() * 1000))
    return secure_filename(now + " " + filename)

# =================================================================

@app.route('/')
def index():
    return render_template('index.jinja2', klasifikasi=True)

@app.route('/upload_csv')
def upload_csv():
    return render_template('upload_csv.jinja2', upload_csv=True)

@app.route('/hasil_upload_csv', methods=['POST'])
def hasil_upload_csv():
    return render_template('hasil_upload_csv.jinja2', upload_csv=True)

@app.route('/evaluasi')
def evaluasi():
    return render_template('evaluasi.jinja2', evaluasi=True)

@app.route('/hasil_evaluasi', methods=['POST'])
def hasil_evaluasi():
    return render_template('hasil_evaluasi.jinja2', evaluasi=True)

if __name__ == "__main__":
    app.run(debug=True)