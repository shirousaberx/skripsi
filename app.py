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
    if 'email' in session:
        if session['role'] == 'Administrator':
            return redirect('/manage_produk')
        else:
            return redirect('/list_sewa')
    else:
        return redirect('/login')