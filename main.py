import cv2
import mediapipe as mp
import time
import tkinter as tk
from multiprocessing import Process

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_landmarker = PoseLandmarker.create_from_options(options)

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

baseline = None
bad_posture_start = None
alert_process = None

BAD_POSTURE_THRESHOLD = 0.15
ALERT_TIME = 15


def landmarks_are_visible(
    nose,
    left_shoulder,
    right_shoulder,
    threshold=0.6
):
    return (
        nose.visibility >= threshold
        and left_shoulder.visibility >= threshold
        and right_shoulder.visibility >= threshold
    )


def draw_point(frame, landmark, color=(0, 255, 0)):
    height, width, _ = frame.shape

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    cv2.circle(frame, (x, y), 7, color, -1)

    return x, y


def get_posture_values(
    nose,
    left_shoulder,
    right_shoulder
):
    shoulder_mid_y = (
        left_shoulder.y + right_shoulder.y
    ) / 2

    shoulder_width = abs(
        left_shoulder.x - right_shoulder.x
    )

    # Avoid division by zero
    if shoulder_width < 0.001:
        return None

    nose_offset_y = (
        nose.y - shoulder_mid_y
    ) / shoulder_width

    shoulder_tilt = (
        left_shoulder.y - right_shoulder.y
    ) / shoulder_width

    return {
        "nose_offset_y": nose_offset_y,
        "shoulder_tilt": shoulder_tilt
    }


def posture_difference(current, baseline):
    y_diff = abs(
        current["nose_offset_y"]
        - baseline["nose_offset_y"]
    )

    tilt_diff = abs(
        current["shoulder_tilt"]
        - baseline["shoulder_tilt"]
    )

    return y_diff + tilt_diff


def show_shrimp_alert():
    root = tk.Tk()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    alert_height = screen_height // 2

    root.geometry(
        f"{screen_width}x{alert_height}+0+0"
    )

    root.overrideredirect(True)
    root.attributes("-topmost", True)

    root.configure(bg="black")

    label = tk.Label(
        root,
        text="SHRIMP ALERT",
        font=("Arial", 60, "bold"),
        fg="red",
        bg="black"
    )

    label.pack(expand=True)

    root.bind(
        "<Escape>",
        lambda event: root.destroy()
    )

    root.mainloop()


def close_alert():
    global alert_process

    if (
        alert_process is not None
        and alert_process.is_alive()
    ):
        alert_process.terminate()
        alert_process = None


def main():
    global baseline
    global bad_posture_start
    global alert_process

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        print("Could not open webcam")
        return

    start_time = time.time()

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        result = pose_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        current_values = None

        # -----------------------------
        # POSE DETECTED
        # -----------------------------
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            nose = landmarks[NOSE]
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]

            nose_point = draw_point(
                frame,
                nose
            )

            left_point = draw_point(
                frame,
                left_shoulder
            )

            right_point = draw_point(
                frame,
                right_shoulder
            )

            cv2.line(
                frame,
                left_point,
                right_point,
                (0, 255, 0),
                3
            )

            # -----------------------------
            # LANDMARKS VISIBLE
            # -----------------------------
            if landmarks_are_visible(
                nose,
                left_shoulder,
                right_shoulder
            ):
                current_values = get_posture_values(
                    nose,
                    left_shoulder,
                    right_shoulder
                )

                # -----------------------------
                # NOT CALIBRATED YET
                # -----------------------------
                if baseline is None:
                    bad_posture_start = None
                    close_alert()

                    cv2.putText(
                        frame,
                        "Press C to calibrate",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                # -----------------------------
                # CALIBRATED
                # -----------------------------
                elif current_values is not None:
                    difference = posture_difference(
                        current_values,
                        baseline
                    )

                    is_bad = (
                        difference
                        > BAD_POSTURE_THRESHOLD
                    )

                    # -----------------------------
                    # BAD POSTURE
                    # -----------------------------
                    if is_bad:
                        if bad_posture_start is None:
                            bad_posture_start = time.time()

                        bad_duration = (
                            time.time()
                            - bad_posture_start
                        )

                        cv2.putText(
                            frame,
                            "Posture: BAD",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Bad for: {bad_duration:.1f}s",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                        if bad_duration >= ALERT_TIME:
                            cv2.putText(
                                frame,
                                "SHRIMP ALERT!",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                3
                            )

                            if (
                                alert_process is None
                                or not alert_process.is_alive()
                            ):
                                alert_process = Process(
                                    target=show_shrimp_alert
                                )

                                alert_process.start()

                    # -----------------------------
                    # GOOD POSTURE
                    # -----------------------------
                    else:
                        bad_posture_start = None
                        close_alert()

                        cv2.putText(
                            frame,
                            "Posture: GOOD",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                    cv2.putText(
                        frame,
                        f"Difference: {difference:.3f}",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

            # -----------------------------
            # LANDMARKS NOT VISIBLE ENOUGH
            # -----------------------------
            else:
                bad_posture_start = None
                close_alert()

                cv2.putText(
                    frame,
                    "Posture: NOT DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

        # -----------------------------
        # NO POSE DETECTED AT ALL
        # -----------------------------
        else:
            bad_posture_start = None
            close_alert()

            cv2.putText(
                frame,
                "Posture: NOT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            "Posture Tracker",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if (
            key == ord("c")
            and current_values is not None
        ):
            baseline = current_values.copy()
            bad_posture_start = None
            close_alert()

            print("Posture calibrated!")
            print(baseline)

        if key == ord("q"):
            break

    close_alert()

    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()


if __name__ == "__main__":
    main()
