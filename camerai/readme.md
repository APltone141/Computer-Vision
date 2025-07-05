# 🎥 CameraAI (camerai)
🚀 AI & Computer Vision toolkit untuk meningkatkan kemampuan webcam murah agar setara fitur premium.

---

## 📌 Fitur Utama
- ✅ **Software Auto-focus** berbasis deteksi wajah/objek
- ✅ **Auto-tracking** wajah atau objek, kamera secara virtual mengikuti target
- ✅ **Motion detection** mendeteksi aktivitas gerak di frame
- ✅ **Gesture control** untuk kendali fitur via gerakan tangan
- ✅ **Real-time resolution enhancement** (misalnya pakai Real-ESRGAN)
- ✅ **FPS enhancement** (interpolasi video real-time/semi-realtime menggunakan RIFE atau DAIN)
- ✅ GUI sederhana untuk user-friendly experience

---

## ⚙️ Teknologi & Library
- **Python** (>=3.9)
- [OpenCV](https://opencv.org/) - image & video processing
- [MediaPipe](https://mediapipe.dev/) / dlib - face & hand tracking
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - super resolution
- [RIFE](https://github.com/megvii-research/RIFE) atau [DAIN](https://github.com/baowenbo/DAIN) - frame interpolation
- GUI: PyQt5 / Tkinter / Streamlit
- Utilitas: numpy, threading, PyInstaller untuk build executable

---

## 🏗 Struktur Project
cameraai/
├── camera_handler/ # Modul pengelola webcam & video capture
├── modules/ # Semua fitur utama dalam modul terpisah
├── gui/ # Script GUI
├── utils/ # Helper untuk performance & logging
├── tests/ # Unit test modul
├── assets/ # Contoh video, image, untuk tes
├── main.py # Entry point
├── requirements.txt # List dependensi
├── README.md
├── .gitignore


---

## 🚀 Cara Instalasi
1️⃣ **Clone repository**
```bash
git clone https://github.com/username/camerai.git
cd camerai

---

## 2️⃣ Install dependensi

Disarankan menggunakan virtualenv / conda

```bash
pip install -r requirements.txt