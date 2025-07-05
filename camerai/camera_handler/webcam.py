import cv2
from threading import Thread, Event
from queue import Queue

class FrameGrabber:
    def __init__(self, src=0, width=640, height=480, queue_size=64):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.queue = Queue(maxsize=queue_size)
        self.stopped = Event()

    def start(self):
        Thread(target=self._capture_loop, daemon=True).start()
        return self

    def _capture_loop(self):
        while not self.stopped.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.queue.full():
                self.queue.get()
            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped.set()
        self.cap.release()