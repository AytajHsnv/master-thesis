import time
import argparse
import codecs
import yaml
from tqdm import tqdm
from newloader import *
from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model.TransMUNet import TransMUNet
from utils.utils import get_img_patches, merge_pred_patches


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='./results.prf')
parser.add_argument('--thresh_step', type=float, default=0.01)
args = parser.parse_args()

def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []

    for thresh in np.arange(0.0, 1.0, thresh_step):
        # print(thresh)
        statistics = []
        statis = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img   = (gt).astype('uint8')
            pred_img = (pred > thresh).astype('uint8')
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))
            # metrics for each image
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
            r = tp/(tp+fn)
            if p+r==0:
                f1=0
            else:
                f1 = 2*p*r/(p+r)
            # calculation IoU
            IoU = tp/(tp+fn+fp)
            statis.append([p, r, f1, IoU])
            
        
        save_result_for_each_img(statis)
        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        # calculate precision
        p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
        # calculate recall
        r_acc = tp/(tp+fn)
        # calculate f-score
        if p_acc+r_acc==0:
            f1_acc=0
        else:
            f1_acc=2*p_acc*r_acc/(p_acc+r_acc)
        # calculation IoU
        IoU = tp/(tp+fn+fp)
        final_accuracy_all.append([thresh, p_acc, r_acc, f1_acc, IoU])
    return final_accuracy_all

def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def save_results(input_list, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        fout.write("Threshold\tPrecision\tRecall\tF1\tIoU\n")
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)

def save_result_for_each_img(img_statistics):
    with codecs.open("each_image_stat.txt", 'w', encoding='utf-8') as fout:
        fout.write("\t\t\tPrecision\tRecall\tF1\tIoU\n")
        for idx, img in enumerate(img_statistics):
            line = f"Image {idx+1}:\t" + '\t'.join(['%.4f'%s for s in img])+'\n'
            fout.write(line)

def save_sample(img_path, msk, msk_pred, name=''):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    msk = msk.astype(int)
    mskp = msk_pred
    _, axs = plt.subplots(1, 3, figsize=(15,5))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].imshow(img/255.)

    axs[1].axis('off')
    axs[1].imshow(msk*255, cmap= 'gray')

    axs[2].axis('off')
    axs[2].imshow(mskp*255, cmap= 'gray')

    plt.savefig(config['save_result'] + name + '.png')

config         = yaml.load(open('./config_crack.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = config['path_to_testdata']
DIR_IMG  = os.path.join(data_path)
DIR_MASK = os.path.join(data_path)
img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
mask_names = [path.name for path in Path(DIR_MASK).glob('*.png')]
img_names.sort()
mask_names.sort()
test_dataset = Crack_loader(img_dir=DIR_IMG, img_fnames=img_names, mask_dir=DIR_MASK, mask_fnames=mask_names)
test_loader  = DataLoader(test_dataset, batch_size = 1, shuffle= False)
print(f'test_dataset:{len(test_dataset)}')

Net = TransMUNet(n_classes = number_classes)
Net = Net.to(device)
Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])

pred_list = []
gt_list = []
save_samples = True # if save_samples=Flase, no samples will be saved.

with torch.no_grad():
    print('val_mode')
    val_loss = 0
    times =0
    Net.eval()

    for itter, batch in enumerate(tqdm(test_loader)):
        img = batch['image'].numpy().squeeze(0)
        img_path = batch['img_path'][0]
        print(img_path)
        msk = batch['mask']
        patch_totensor = ImgToTensor()
        preds = []
            
        start = time.time()
        patches, patch_locs = get_img_patches(img)
        for i, patch in enumerate(patches):
            patch_n = patch_totensor(Image.fromarray(patch))         # torch.Size([3, 256, 256])
            X = (patch_n.unsqueeze(0)).to(device, dtype=torch.float) # torch.Size([1, 3, 256, 256])
            msk_pred = torch.sigmoid(Net(X))                         # torch.Size([1, 1, 256, 256])
            mask = msk_pred.cpu().detach().numpy()[0, 0]             # (256, 256)
            preds.append(mask)
        mskp = merge_pred_patches(img, preds, patch_locs)            # (H, W)
        kernel = np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ], dtype=np.uint8)
        mskp = cv2.morphologyEx(mskp, cv2.MORPH_CLOSE, kernel,iterations=1).astype(float)
        end = time.time()
        times += (end - start)
        if itter < 237 and save_samples:
            save_sample(img_path, msk.numpy()[0, 0], mskp, name=str(itter+1))

        gt_list.append(msk.numpy()[0, 0])
        pred_list.append(mskp)
    print('Running time of each images: %ss' % (times/len(pred_list)))

final_results = []
final_results = cal_prf_metrics(pred_list, gt_list, args.thresh_step)
save_results(final_results, args.output)