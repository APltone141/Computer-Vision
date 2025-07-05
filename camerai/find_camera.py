import cv2

def list_available_cameras(max_index=10):
    print(" Mendeteksi kamera yang tersedia...")
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f" Kamera ditemukan di index: {index}")
            cap.release()
        else:
            print(f" Tidak ada kamera di index: {index}")
        cap.release()

if __name__ == "__main__":
    list_available_cameras()
