import glob
import os
from concurrent.futures import ProcessPoolExecutor

source_dir = "/project/dane2/wficai/pda/external_validation/Boston/exports/"
target_dir = "/project/dane2/wficai/pda/external_validation/Boston/frames/"

# recursively get all the mpa4 files in the root directory
mp4_files = glob.glob(source_dir + "/**/*.mp4", recursive=True)

def convert_mp4_to_jpgfolder(mp4_path):
    # get the name of the file without the extension
    study_folder = mp4_path.split("/")[-2]
    file_name = mp4_path.split("/")[-1].split(".mp4")[0]
    # create a folder with the same name as the file
    vid_folder = target_dir + study_folder + '/' + file_name
    os.makedirs(vid_folder, exist_ok=True)

    # convert the mp4 file to jpg files
    os.system(f"ffmpeg -i {mp4_path} -vf fps=30 {vid_folder}/frame_%04d.jpg")

with ProcessPoolExecutor() as executor:
    executor.map(convert_mp4_to_jpgfolder, mp4_files)



