from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ TEST ROUTE UNTUK CEK SERVER
@app.route('/', methods=['GET'])
def index():
    return '✅ NusaMiner Flask backend is live!'

# 🔧 CONTOH ENDPOINT LAINNYA (hapus jika tak digunakan)
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'})

# 🧠 Tambahkan endpoint ML kamu lainnya di bawah ini:
# @app.route('/predict', methods=['POST']) ...
# @app.route('/confusion_matrix', methods=['GET']) ...
# dst

# ✅ WAJIB: Gunakan PORT dari Railway
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
