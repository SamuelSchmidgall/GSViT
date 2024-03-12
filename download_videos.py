import os, cv2


if __name__ == "__main__":
	dir_path = "videos/"
	files = os.listdir(dir_path)

	video_list = list()
	surgical_videos = dict()
	video_words = list()
	video_frames = {}
	framesps = {}
	
	total_words = 0
	total_frames = 0
	total_videos = 0
	total_captions = 0
	for _file in files:
		_file_path = dir_path + _file
		if ".mp4" in _file_path:
			total_videos += 1
			cap = cv2.VideoCapture(_file_path)
			total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			frame = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			fps = int(cap.get(cv2.CAP_PROP_FPS))
			if frame in video_frames:
				video_frames[frame] += 1
			else:
				video_frames[frame] = 1
			if fps in framesps:
				framesps[fps] += 1
			else:
				framesps[fps] = 1
			cap.release()
		if ".vtt" in _file_path:
			total_captions += 1
			words_in_video = 0
			with open(_file_path) as f:
				text_data = f.readlines()
				for line in text_data:
					if len(line) > 0 and line[0] != 0:
						total_words += len(line.split(" "))
						words_in_video += len(line.split(" "))
			video_words.append(words_in_video)
		#print("Total frames:", total_frames, "Total words:", total_words)
	#print(framesps)
	#print(video_frames)
	resolutions = []
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
			
	import numpy as np
	from pytube import YouTube 
	total_seconds = 0
	seconds_list = []
	unique_authors = list()
	for _video in video_list:
		try:
		    yt = YouTube(_video)
		    total_seconds += yt.length
		    author = yt.author
		    if author not in unique_authors:
			    unique_authors.append(author)
		    seconds_list.append(yt.length)
		except Exception: pass
		#print("Total time: ", total_seconds/(60**2))
	
	print("~"*50)
	video_words = np.array(video_words)
	seconds_list = np.array(seconds_list)
	print("Videos: ", total_videos, "Captions: ", total_captions)
	print("Words mean: ", np.mean(video_words), "Words STD: ", np.std(video_words))
	print("Seconds MEAN, STD: ", np.mean(seconds_list), np.std(seconds_list))
	print("Seconds MAX, MIN: ", np.max(seconds_list), np.min(seconds_list))
	print("Total time: ", total_seconds/(60**2))
	print("Total frames:", total_frames, "Total words:", total_words)
	print("Unique Surgeons: ", len(unique_authors))
		
		
					

	
