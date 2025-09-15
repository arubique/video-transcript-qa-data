import os
import wget
import json
from tqdm import tqdm
import argparse

ROOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(ROOT_PATH, "data", "input_videos")
# VIDEO_DIR_NAME = "opencourseware"


# code is adapted from: https://github.com/repetitioestmaterstudiorum/lecture-video-rag/blob/master/dev/rag_example.ipynb
def main():
    parser = argparse.ArgumentParser(
        description="Download example videos from JSON file"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=os.path.join(DATA_DIR, "example_videos.json"),
        help="Path to JSON file containing video URLs and metadata",
    )
    parser.add_argument(
        "--video_dir_name",
        type=str,
        default="opencourseware",
        help="Name of the video directory",
    )
    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        videos_to_download = json.load(f)["videos"]

    for video in (pbar := tqdm(videos_to_download)):
        video_folder = os.path.join(
            DATA_DIR, args.video_dir_name, video["directory"]
        )
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        if not os.path.exists(os.path.join(video_folder, video["filename"])):
            pbar.set_description(
                f"Downloading {video['filename']} to {video_folder}"
            )
            wget.download(
                video["url"], out=os.path.join(video_folder, video["filename"])
            )


if __name__ == "__main__":
    main()
