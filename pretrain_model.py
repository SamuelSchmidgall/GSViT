import cv2
import time
import torch
import pickle
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torchvision.transforms import ToPILImage
from EfficientViT.classification.model.build import EfficientViT_M5


class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Decoder(nn.Module):
    def __init__(self, in_size, predict_change=False):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.predict_change = predict_change

        # Initial representation
        self.fc = nn.Linear(384*4*4, 7 * 7 * 1024)
        self.bn1d = nn.BatchNorm1d(7 * 7 * 1024)
        self.gelu = nn.GELU()

        # Decoder layers
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()
        #self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv2 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, output_padding=0)
        #self.bn2 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU()
        #self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        #self.bn3 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        #self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        #self.bn4 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0)

        # Residual blocks with SE attention
        self.res2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            SEAttention(64),
            nn.ReLU()
        )

        # was 256
        self.res1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            SEAttention(512),
            nn.ReLU()
        )
        if not self.predict_change:
            self.sigmoid = nn.Sigmoid()
        else:
            self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x.reshape(self.in_size, 384*4*4))
        x = self.bn1d(x)
        x = self.gelu(x)
        x = x.view(-1, 1024, 7, 7)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.res1(x) + x
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.res2(x) + x
        x = self.conv5(x)
        if not self.predict_change:
            x = self.sigmoid(x)
        else:
            x = self.tanh(x)
        return x


class EfficientViTAutoEncoder(nn.Module):
    def __init__(self, in_size, predict_change=False):
        super(EfficientViTAutoEncoder, self).__init__()
        self.predict_change = predict_change
        self.decoder = Decoder(in_size, predict_change)
        self.evit = EfficientViT_M5(pretrained='efficientvit_m5')
        # remove the classification head
        self.evit = torch.nn.Sequential(*list(self.evit.children())[:-1])

    def forward(self, x):
        out = self.evit(x)
        decoded = self.decoder.forward(out)
        return decoded



if __name__ == "__main__":
    # linear schedule
    epochs = 5
    in_size = 100
    batch_size = 100
    data_processed = 0
    num_data = 70000000//2
    itrs_per_epoch = num_data//batch_size

    lr_start = 1e-3
    lr_finish = 1e-5

    lr = lr_start
    step = (lr_start - lr_finish)/(itrs_per_epoch*epochs)

    gpu_parallel = False
    data_parallel = True
    custom_dataset = True
    predict_change = False

    num_devices = 1
    if gpu_parallel: 
        num_devices = torch.cuda.device_count()
        num_devices = num_devices if num_devices > 0 else 1
    in_size, batch_size = in_size//num_devices, batch_size//num_devices

    evitae = EfficientViTAutoEncoder(in_size, predict_change=predict_change)
    if gpu_parallel and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPU(s)")
        evitae = nn.DataParallel(evitae)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    evitae.to(device)


    running_loss = 0
    running_loss_lg = 0
    epoch_est_filter = 0
    optim = torch.optim.Adam(evitae.parameters())

    evitae.train()
    if custom_dataset:
        from dataloader_surgical import load_data
        dataset = load_data(
            num_images=batch_size, 
            data_root="../surgical_simvp/data/", 
            num_workers=1,
            predict_change=predict_change,)
        dataset.parallel_generate()
    else:
        images = torch.rand(1, 3, 224, 224).repeat(in_size*num_devices, 1, 1, 1)
        images = images.to(device=device)

    for _epoch_itr in range(epochs):
        for _itr in range(itrs_per_epoch):
            itr_start = time.time()
            reconstruct_loss = 0
            # generate minibatch indices
            minibatches = torch.arange(batch_size)
            idx = torch.randperm(minibatches.shape[0])
            minibatches = minibatches[idx].view(minibatches.size())
            minibatches = list(torch.split(minibatches, in_size))
            # iterate through minibatch indices
            for mb_ind in range(len(minibatches)):
                data_processed += in_size
                if custom_dataset:
                    images, targets = dataset.get(minibatches[mb_ind])
                    if gpu_parallel:
                        images = images.to(device)
                        targets = targets.to(device)
                else:
                    target = images
                decoded = evitae(images)
                _sub_rcloss = abs(decoded.flatten() - targets.detach().flatten()).mean()
                reconstruct_loss += _sub_rcloss

            reconstruct_loss.backward()
            optim.step()

            lr -= step
            for _g in optim.param_groups:
                _g['lr'] = lr

            optim.zero_grad()
            t = time.time()
            dataset.generate_dataset(parallel_call=True)

            filter_res = 0.99
            filter_run_long = 0.997
            filter_run_short = 0.98
            itr_end = time.time()
            print_loss = reconstruct_loss.detach().cpu().numpy()
            running_loss = filter_run_short*running_loss + (1-filter_run_short)*print_loss
            running_loss_lg = filter_run_long*running_loss_lg + (1-filter_run_long)*print_loss
            epoch_est_filter = filter_res*epoch_est_filter + \
                (1-filter_res)*(num_data*((itr_end - itr_start)/batch_size))/3600
        
            if _itr == 0:
                running_loss = print_loss
                running_loss_lg = print_loss
                epoch_est_filter = (num_data*((itr_end - itr_start)/batch_size))/3600
            
            if (_itr+1) % 10 == 0:
                print("~~~"*30,
                    "\nData Processed:", data_processed*2,
                    "\nIteration:", "{}/{} = {}%".format(_itr, itrs_per_epoch, round((_itr/itrs_per_epoch)*100, 5), "Epoch", _epoch_itr),
                    "\nTime:", round(itr_end - itr_start, 4), "s",
                    "\nEst Time Per Epoch:", epoch_est_filter,
                    "\nEst Time Left in Epoch:", (1-round(_itr/itrs_per_epoch, 5))*epoch_est_filter,
                    "\nLoss: {}, Running Loss: {}:".format(round(float(running_loss), 5), round(float(running_loss_lg), 5)),)
            
            if (_itr+1) % 1000 == 0:
                #with open("evit_save/saved_network{}_{}.pkl".format(_epoch_itr, _itr//1000), "wb") as f:
                #    pickle.dump(evitae.state_dict(), f)
                torch.save(evitae.state_dict(), "evit_train2/saved_network{}_{}.pkl".format(_epoch_itr, _itr//1000))
            if (_itr+1) % 50 == 0:
                first_image = images[0].cpu()
                #first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
                first_decoded = decoded[0].cpu()
                #first_decoded = cv2.cvtColor(decoded[0].cpu(), cv2.COLOR_BGR2RGB)
                first_tensor = torch.cat((first_image, first_decoded), dim=2)
                first_tensor = first_tensor / first_tensor.max()
                to_pil = ToPILImage()
                image = to_pil(first_tensor)
                image.save('output_image.png')


