import cv2, sys, os
from yt_dlp import YoutubeDL


os.mkdir("videos/")
save_videos_path = "videos/"
dir_path = "videos_to_download/"
files = os.listdir(dir_path)
video_list = list()
surgical_videos = dict()
for _file in files:
    _file_path = dir_path + _file
    with open(_file_path, "r") as f:
        links = f.readlines()
        links = [_.replace("\n", "") for _ in links if len(_) > 1]
        surgical_videos[_file.replace(".txt", "")] = links
        video_list += links

existing = []
files = os.listdir(save_videos_path)
for _file in files:
    _file_path = _file.split(".")[0]
    existing.append(_file_path)

total_seconds = 0
seconds_list = []
unique_authors = list()
for _video in video_list:
    try:
        print(_video)
        capid = _video.split("/")[-1]
        if capid in existing: continue
        ydl_opts = {
            'outtmpl': 'videos/{}.%(title)s.%(ext)s'.format(capid),
            'writesubtitles': True,
            'subtitle': '--write-sub --sub-lang en',
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([_video])

    
    except Exception as e: 
        print(e, _video)
    #print("Total time: ", total_seconds/(60**2))

print("~"*50)
seconds_list = np.array(seconds_list)

		
					

	
