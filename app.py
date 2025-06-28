from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return '✅ NusaMiner Flask backend is live!'

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'})

# ⚠️ JANGAN PAKAI app.run() jika pakai Gunicorn!
# Gunicorn akan otomatis men-deploy object `app`

# (Tidak perlu blok `if __name__ == "__main__"` di bawah ini)
