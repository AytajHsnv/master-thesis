from __future__ import division
import copy
import torch.optim as optim
from utils.utils import *
from pathlib import Path
from datetime import datetime
from newloader import Crack_loader
from model.TransMUNet import TransMUNet
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


setup_seed(42)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

data_tra_path = config['path_to_tradata']
data_val_path = config['path_to_valdata']

# root_path='./Cityscapes'
# DIR_IMG_tra  = os.path.join(root_path, 'train', 'img')
# DIR_MASK_tra = os.path.join(root_path, 'train', 'label')
# print(DIR_IMG_tra)

# DIR_IMG_val  = os.path.join(root_path, 'val', 'img')
# DIR_MASK_val = os.path.join(root_path, 'val', 'label')
# print(DIR_IMG_val)

DIR_IMG_tra  = os.path.join(data_tra_path)
DIR_MASK_tra = os.path.join(data_tra_path)
print(DIR_IMG_tra)

DIR_IMG_val  = os.path.join(data_val_path)
DIR_MASK_val = os.path.join(data_val_path)

img_names_tra  = [path.name for path in Path(DIR_IMG_tra).glob('*.jpg')]
mask_names_tra = [path.name for path in Path(DIR_MASK_tra).glob('*.png')]
print(img_names_tra)

img_names_val  = [path.name for path in Path(DIR_IMG_val).glob('*.jpg')]
mask_names_val = [path.name for path in Path(DIR_MASK_val).glob('*.png')]


train_dataset = Crack_loader(img_dir=DIR_IMG_tra, img_fnames=img_names_tra, mask_dir=DIR_MASK_tra, mask_fnames=mask_names_tra, isTrain=True)
valid_dataset = Crack_loader(img_dir=DIR_IMG_val, img_fnames=img_names_val, mask_dir=DIR_MASK_val, mask_fnames=mask_names_val, resize=True)


print(f'train_dataset:{len(train_dataset)}')
print(f'valiant_dataset:{len(valid_dataset)}')


# train_loader  = DataLoader(train_dataset, batch_size = int(config['batch_size_tr']), shuffle= True,  drop_last=True)
# print(train_loader)
# val_loader    = DataLoader(valid_dataset, batch_size = int(config['batch_size_va']), shuffle= False, drop_last=True)
train_loader  = DataLoader(train_dataset, batch_size = 4, shuffle= True,  drop_last=True)
print(train_loader)
val_loader    = DataLoader(valid_dataset, batch_size = 1, shuffle= False, drop_last=True)
Net = TransMUNet(n_classes = number_classes)
flops, params = get_model_complexity_info(Net, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
print('flops: ', flops, 'params: ', params)
message = 'flops:%s, params:%s' % (flops, params)

Net = Net.to(device)
# load pretrained model
if int(config['pretrained']):
    # pretrained_model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    # pretrained_dict = pretrained_model.state_dict()
    # model_dict = Net.state_dict()
    # # Filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    # print('Pretrained dict:', pretrained_dict)
    # # Update your model's state_dict
    # Net.load_state_dict(pretrained_dict)
   

    Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
    best_val_loss = torch.load(config['saved_model'], map_location='cpu')['val_loss']
optimizer = optim.AdamW(Net.parameters(), lr= float(config['lr']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = config['patience'])
criteria  = DiceBCELoss()


# visual
visualizer = Visualizer(isTrain=True)
log_name = os.path.join('./checkpoints', config['loss_filename'])
with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)

epoch_losses = []
for ep in range(int(config['epochs'])):
    # train
    Net.train()
    epoch_loss = 0
    for itter, batch in enumerate(train_loader):
        # img,mask = batch
        # img = img.to(device, dtype=torch.float)
        # msk = mask.to(device)
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask'].to(device)

        
        boundary = batch['boundary'].to(device)
        mask_type = torch.float32 if Net.n_classes == 1 else torch.long
        msk = msk.to(device=device, dtype=mask_type)
        boundary = boundary.to(device=device, dtype=mask_type)
        msk_pred, B = Net(img,istrain=True)
        if(msk.shape != msk_pred.shape):
             msk = msk[:,:,:,:,0]
        loss = criteria(msk_pred, msk)
        if B.shape != boundary.shape:
            boundary=boundary[:,:,:,:,0]
        loss_boundary = criteria(B, boundary)
        tloss    = (0.8*(loss)) + (0.2*loss_boundary) 
        optimizer.zero_grad()
        tloss.backward()
        epoch_loss += tloss.item()
        
        optimizer.step()  
        if (itter+1)%int(float(config['progress_p']) * len(train_loader)) == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f' Epoch>> {ep+1} and iteration {itter+1} loss>>{epoch_loss/(itter+1)}')
        if (itter+1)*int(config['batch_size_tr']) == len(train_dataset):
            visualizer.print_current_losses(epoch=(ep+1), iters=(itter+1), loss=((epoch_loss/(itter+1))), lr=lr, isVal=False)
    epoch_losses.append(epoch_loss)

    # eval        
    with torch.no_grad():
        print('val_mode')
        val_loss = 0
        Net.eval()
        for itter, batch in enumerate(val_loader):
            # img,mask = batch
            # img = img.to(device, dtype=torch.float)
            # msk = mask.to(device)
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            mask_type = torch.float32 if Net.n_classes == 1 else torch.long
            msk = msk.to(device=device, dtype=mask_type)
            msk_pred = Net(img)
            if msk.shape != msk_pred.shape:
                msk = msk[:,:,:,:,0]
            loss = criteria(msk_pred, msk)
            val_loss += loss.item()
        visualizer.print_current_losses(epoch=ep+1, loss=(abs(val_loss/(itter+1))), isVal=True)   
        mean_val_loss = (val_loss/(itter+1))
        # Check the performance and save the model
        if mean_val_loss < best_val_loss:
            best = ep + 1
            best_val_loss = copy.deepcopy(mean_val_loss)
            print('New best loss, saving...,best_val_loss=%6f' % (best_val_loss))
            with open(log_name, "a") as log_file:
                message = 'New best loss, saving...,best_val_loss=%6f' % (best_val_loss)
                log_file.write('%s\n' % message)
            state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})
            torch.save(state, config['saved_model'])

    scheduler.step(mean_val_loss)

    if ep+1 == int(config['epochs']): 
        visualizer.print_end(best, best_val_loss)
        state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})
        torch.save(state, config['saved_model_final'])

print('Training phase finished')    
plt.plot(range(int(config['epochs'])), epoch_losses)
plt.show()
current_datetime_losses = datetime.now().strftime("%Y-%m-%d")
plt.savefig(current_datetime_losses)