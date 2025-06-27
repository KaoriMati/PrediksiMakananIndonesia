from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO
import numpy as np
import json


# Inisialisasi Flask
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model_makanan_indo.h5")

# Load data nutrisi
with open("nutrition_info.json", "r", encoding="utf-8") as f:
    nutrition_info = json.load(f)

# Daftar label kelas (urutan sesuai model training)
class_labels = [
    'Ayam Goreng', 'Burger', 'French Fries', 'Gado-Gado', 'Ikan Goreng',
    'Mie Goreng', 'Nasi Goreng', 'Nasi Padang', 'Pizza', 'Rawon',
    'Rendang', 'Sate', 'Soto'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'File gambar tidak ditemukan'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
    try:
        # Baca dan konversi gambar
        img = load_img(BytesIO(file.read()), target_size=(224, 224))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Prediksi
        preds = model.predict(x, verbose=0)
        idx = np.argmax(preds[0])
        label = class_labels[idx]
        akurasi = float(preds[0][idx])

        # Ambil info gizi
        gizi = nutrition_info.get(label, "Informasi gizi tidak tersedia")

        return jsonify({
            'label': label,
            'akurasi': round(akurasi, 4),
            'nutrition': gizi
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
