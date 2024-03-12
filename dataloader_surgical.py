import os
import gzip
import pickle
import random
import time
import torch.nn.functional as F

import numpy as np
import torch
import torch.utils.data as data
from torch.multiprocessing import Process
import torch.multiprocessing as mp #, Queue


from numpy import *
import cv2
from PIL import Image

def read_video(n_frames=None, video_loc=None):
    i = 0
    all = []
    cap = cv2.VideoCapture(video_loc) #"rec_q26b_10min.mp4")
    if n_frames is None:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        arr = np.array(frame)
        all.append(arr)
        i += 1
    return np.array(all)

def load_surgical(root, frames=None):
    videos = list()
    for video in os.listdir(root):
        videos.append(read_video(video_loc=root + "/" + video, n_frames=frames))
    return videos


def read_clips(total_clips_dur, video_loc):
    cap = cv2.VideoCapture(video_loc)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = random.randint(0, n_frames - 1 - total_clips_dur)

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_frames = []
    for _ in range(total_clips_dur):  # Only read the required number of frames
        ret, frame = cap.read()
        if not ret:  # If frame reading was not successful, break
            break
        all_frames.append(np.array(frame))

    cap.release()  # Make sure to release the capture after reading
    return np.array(all_frames)


def lazy_load_surgical(arr):
    clips = list()
    #root="../surgical_simvp/data"
    root="../../Downloads/SurgicalDataset/videos/videos_dwn_sorted/{}/".format("colectomy")
    #videos= os.listdir(root + "/surgical/train/")
    videos= os.listdir(root)
    num_videos=100
    clip_dur=2

    #total_frames=0
    #video_probs = list()
    #for video in videos:
    #    cap = cv2.VideoCapture(root + "/surgical/train/" + video)
    #    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #    total_frames += frames
    #    video_probs.append(frames)
    #for _ in range(len(video_probs)):
    #    video_probs[_] /= total_frames
    sampled_videos = [random.choice(videos,) for _ in range(num_videos)]
    for video in sampled_videos:
        #cap = cv2.VideoCapture(root + "/surgical/train/" + video)
        cap = cv2.VideoCapture(root + video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = random.randint(0, n_frames - 1 - clip_dur)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        all_frames = []
        for _ in range(clip_dur):  # Only read the required number of frames
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(np.array(frame))
        cap.release()
        clip = np.array(all_frames)
        clips.append(torch.tensor(clip, device="cuda:3").unsqueeze(0))
    if arr is not None: 
        arr[:] = torch.cat(clips, dim=0).squeeze()[:].clone()
    else:
        return clips


class SurgicalDataset(data.Dataset):
    def __init__(self, 
                 root, 
                 is_train=True, 
                 n_frames_input=1, 
                 n_frames_output=1, 
                 transform=None,
                 batch_size=128,
                 predict_change=False, 
                 finetune=False):
        super(SurgicalDataset, self).__init__()

        self.root = "../../Downloads/SurgicalDataset/videos/videos_dwn_sorted/{}/".format("colectomy")
        self.dataset = None
        self.finetune = finetune
        self.batch_size = batch_size
        self.predict_change = predict_change
        #if not finetune:
        #    self.videos = os.listdir(root + "surgical/train")
        #else:
        self.videos = os.listdir(root)
        self.num_video = len(self.videos)
        # probability of each video relative to frame count
        total_frames = 0
        video_probs = list()
        for video in self.videos:
            #cap = cv2.VideoCapture(
            #    root + "surgical/train/" if not finetune else "" + video)
            cap = cv2.VideoCapture(
                root + video)
            
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames += frames
            video_probs.append(frames)
        for _ in range(len(video_probs)):
            video_probs[_] /= total_frames
        self.video_probs = video_probs

        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform

        # data format:
        # (time in + time out) X batch X image_x X image_y x 1
        self.clips = list()
        self.generate_dataset()

        self.std = 1
        self.mean = 0
        mp.set_start_method('spawn')

        self.procs = 12
        self.proc_id = 0
        for proc in range(self.procs):
            setattr(self, "parallel_proc{}".format(proc), None)
            setattr(self, "return_arr{}".format(proc),
                    torch.zeros((batch_size, 2, 144, 256, 3), device="cuda:3").share_memory_())

    def parallel_generate(self, proc_id=None):
        if proc_id is not None:
            setattr(self, "parallel_proc{}".format(proc_id), Process(
                target=lazy_load_surgical,
                args=(getattr(self, "return_arr{}".format(proc_id)),)))
            getattr(self, "parallel_proc{}".format(proc_id)).start()
        else:
            for proc in range(self.procs):
                setattr(self, "parallel_proc{}".format(proc), Process(
                    target=lazy_load_surgical,
                    args=(getattr(self, "return_arr{}".format(proc)),)))
                getattr(self, "parallel_proc{}".format(proc)).start()


    def generate_dataset(self, parallel_call=False):
        """
        We want to take random clip segments from videos
        todo: randomize the video "speed" interpolating between frames?
        """
        if parallel_call:
            # wait for process to return dataset
            getattr(self, "parallel_proc{}".format(self.proc_id)).join()
            # get return array from parallel process
            self.clips = getattr(self, "return_arr{}".format(self.proc_id)).clone()
            # calculate length
            self.length = self.clips.shape[0]
            # regenerate process
            self.parallel_generate(proc_id=self.proc_id)
            # set pointer to next process
            self.proc_id = (self.proc_id + 1) % self.procs
            return
        lazy_load_dataset = lazy_load_surgical(arr=None)
        self.clips = lazy_load_dataset
        self.clips = [torch.tensor(_, device="cuda:3").unsqueeze(0).clone().detach() for _ in self.clips]
        self.clips = torch.cat(self.clips, dim=0)
        self.length = len(self.clips)

    def __len__(self):
        return self.length

    def get(self, idx):
        # Define the resize transformation
        clips = self.clips[idx].squeeze()
        inp = (clips[:, 0:1, :, :, :] / 255.0).contiguous().float().squeeze().permute(0, 2, 3, 1).permute(0, 2, 3, 1)
        out = (clips[:, 1:2, :, :, :] / 255.0).contiguous().float().squeeze().permute(0, 2, 3, 1).permute(0, 2, 3, 1)
        inp = F.interpolate(inp, size=(224, 224), mode='bilinear', align_corners=False)
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        if self.predict_change:
            out = out - inp
        return inp, out


def load_data(num_images, data_root, num_workers, predict_change=False):
    train_set = SurgicalDataset(
        root=data_root, is_train=True, batch_size=num_images,
        n_frames_input=1, n_frames_output=1, predict_change=predict_change)
    return train_set

def finetune_data(num_images, data_root, num_workers, predict_change=False):
    train_set = SurgicalDataset(
        root=data_root, is_train=True, batch_size=num_images, finetune=True,
        n_frames_input=1, n_frames_output=1, predict_change=predict_change)
    return train_set

if __name__ == "__main__":
    dataloader_train = load_data(10000, 1, "./data/", 1)
    import nvsmi
    print(nvsmi.get_gpu_processes())
