import cv2
import os

def extract_frames(video_path, output_folder):

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # extract every 15 frames
        if frame_count % 15 == 0:
            filename = os.path.join(
                output_folder,
                f"{os.path.basename(video_path)}_{frame_count}.jpg"
            )
            cv2.imwrite(filename, frame)

        frame_count += 1

    cap.release()


def process_folder(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video in os.listdir(input_folder):

        video_path = os.path.join(input_folder, video)

        if video.endswith(".mp4"):
            extract_frames(video_path, output_folder)


# Update paths according to your screenshot

process_folder(
    r"C:\Users\deepa\Downloads\archive\Celeb-real",
    r"../dataset/real"
)

process_folder(
    r"C:\Users\deepa\Downloads\archive\Celeb-synthesis",
    r"../dataset/fake"
)

process_folder(
    r"C:\Users\deepa\Downloads\archive\YouTube-real",
    r"../dataset/real"
)