from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
from io import BytesIO
import numpy as np
from PIL import Image
import scipy.fftpack as fft
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.exceptions import PydubException
from scipy.fftpack import dct, idct
from scipy.io import wavfile


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Menetapkan direktori unggahan
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  #
# Set path ffmpeg secara eksplisit
AudioSegment.converter = "C:ffmpeg/bin/ffmpeg.exe"  # Ganti dengan lokasi ffmpeg yang sesuai

def compress_image(image):
    image = np.array(image.convert('L'))
    
    # Blok ukuran 8x8
    h, w = image.shape
    h8, w8 = h // 8 * 8, w // 8 * 8
    image = image[:h8, :w8]
    
    # Fungsi DCT dan IDCT
    def dct2(a):
        return fft.dct(fft.dct(a.T, norm='ortho').T, norm='ortho')
    
    def idct2(a):
        return fft.idct(fft.idct(a.T, norm='ortho').T, norm='ortho')
    
    # Terapkan DCT dan IDCT pada blok 8x8
    dct_blocks = np.zeros_like(image, dtype=float)
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = image[i:i+8, j:j+8]
            dct_blocks[i:i+8, j:j+8] = dct2(block)
    
    # Kuantisasi (mengatur koefisien frekuensi tinggi ke 0)
    thresh = 50
    dct_blocks[np.abs(dct_blocks) < thresh] = 0
    
    # Kembali ke domain spasial
    compressed_image = np.zeros_like(image, dtype=float)
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = dct_blocks[i:i+8, j:j+8]
            compressed_image[i:i+8, j:j+8] = idct2(block)
    
    compressed_image = np.clip(compressed_image, 0, 255)
    compressed_image = Image.fromarray(compressed_image.astype(np.uint8))
    return compressed_image

def compress_video(video_path):
    # This is a placeholder for video compression logic
    clip = mp.VideoFileClip(video_path)
    return clip

def dct_compress_audio(file_path):
    # Membaca file audio
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples())

    # Menerapkan DCT
    dct_coefficients = dct(samples, norm='ortho')

    # Menghapus koefisien kecil
    threshold = 0.1 * np.max(dct_coefficients)
    dct_coefficients[np.abs(dct_coefficients) < threshold] = 0

    # Menerapkan inverse DCT
    compressed_signal = idct(dct_coefficients, norm='ortho')
    compressed_signal = np.int16(compressed_signal / np.max(np.abs(compressed_signal)) * 32767)

    # Menyimpan audio yang telah dikompresi
    compressed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_audio.wav')
    wavfile.write(compressed_file_path, audio.frame_rate, compressed_signal)
    
    return compressed_file_path


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    original_size = len(file.read())
    file.seek(0)
    
    image = Image.open(file.stream)
    compressed_image = compress_image(image)
    
    buf = BytesIO()
    compressed_image.save(buf, format='JPEG')
    buf.seek(0)
    
    compressed_data = buf.getvalue()
    compressed_size = len(compressed_data)
    
    return jsonify({
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compressed_image": compressed_data.decode('latin1')
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)
    
    original_size = os.path.getsize(file_path)
    
    compressed_video = compress_video(file_path)
    
    compressed_path = os.path.join("temp", "compressed_" + file.filename)
    compressed_video.write_videofile(compressed_path, codec='libx264')
    
    compressed_size = os.path.getsize(compressed_path)
    
    return jsonify({
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compressed_path": compressed_path
    })

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        compressed_file_path = dct_compress_audio(file_path)
        return send_file(compressed_file_path, as_attachment=True)

@app.route('/download/<file_type>', methods=['POST'])
def download(file_type):
    if file_type == 'image':
        file_data = request.json['compressed_image'].encode('latin1')
        buf = BytesIO(file_data)
        return send_file(buf, mimetype='image/jpeg')
    elif file_type == 'video':
        file_path = request.json['compressed_path']
        return send_file(file_path, mimetype='video/mp4')
    elif file_type == 'audio':
        file_path = request.json['compressed_path']
        return send_file(file_path, mimetype='audio/mpeg')
    else:
        return "Invalid file type", 400

if __name__ == '__main__':
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)
