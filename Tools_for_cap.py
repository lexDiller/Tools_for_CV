import cv2
import os
import glob


def merge_videos(videos_folder, output_file):
    video_files = glob.glob(videos_folder)

    if not video_files:
        print("No videofiles")
        return

    # Take parameters
    cap = cv2.VideoCapture(video_files[0])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create object videoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"Video saves as '{output_file}'.")


def save_video_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f'{output_folder}frame_{frame_number}.jpg'
        cv2.imwrite(frame_filename, frame)

    cap.release()


def resize_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)

            height, width = img.shape[:2]
            if width > 1920 or height > 1080:
                img = cv2.resize(img, (1920, 1080))

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img)
