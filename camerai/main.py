import cv2
from camera_handler.webcam import FrameGrabber
from modules.autofocus import AutoFocus


def main():
    grabber = FrameGrabber(src=1).start()  # ganti src sesuai webcam external
    autofocus = AutoFocus(min_confidence=0.6, padding=50)

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                continue

            output = autofocus.process(frame)

            cv2.imshow("CamerAI - AutoFocus", output)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()