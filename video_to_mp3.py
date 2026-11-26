# convert the videos to mp3
import os
import subprocess

files = os.listdir("videos")
os.makedirs("audios", exist_ok=True)

# Sort videos by lecture number
files = sorted(
    [file for file in files if file.endswith(".mp4")],
    key=lambda file: int(
        file.split("lecture_")[-1].split(".mp4")[0]
    )
)

for file in files:
    lecture_no = int(
        file.split("lecture_")[-1].split(".mp4")[0]
    )

    title = file.split("lecture")[0].replace("_", " ").strip()

    # Display lecture number and title
    print(f"Lecture-{lecture_no:02d}: {title}")

    # Input video path
    input_path = os.path.join("videos", file)

    # Output audio filename with lecture number at the beginning
    output_path = os.path.join(
        "audios",
        f"Lecture-{lecture_no:02d}_{title}.mp3"
    )

    # Convert MP4 to MP3
    subprocess.run(
        [
            "ffmpeg",
            "-i", input_path,
            "-vn",
            "-q:a", "2",
            output_path
        ],
        check=True
    )