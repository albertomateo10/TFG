import cv2
import os

def extract_frames(video_path, output_folder, capture_interval):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Get the frames per second (FPS) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    capture_frame_count = int(fps * capture_interval)
    saved_frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save frame at specified intervals
        if frame_count % capture_frame_count == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved {output_path}")
            saved_frame_count += 1

        frame_count += 1

    video.release()
    print("Process completed.")

# Example usage
video_path = "/DJI_0760.mp4"  # Replace this with the path to your video
output_folder = "/VisDrone/Frames"               # Folder where frames will be saved
capture_interval = 0.5                  # Capture a frame every x seconds
extract_frames(video_path, output_folder, capture_interval)