from datetime import datetime
import time
import argparse
import codecs
import yaml
from tqdm import tqdm
from model import deepLab
from newloader import *
from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model.TransMUNet import TransMUNet
from utils.utils import get_img_patches, merge_pred_patches
from natsort import natsorted
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='./results.prf')
parser.add_argument('--thresh_step', type=float, default=0.01)
args = parser.parse_args()
model = 'DeepLabV3+_MobileNet_Crack500 Labphotos_Gamma'
folder = ['gain', 'gamma', '']
def cal_prf_metrics(pred_list, gt_list, distance=[], angle=[], thresh_step=0.01, img_names=None):
    final_accuracy_all = []
    
     # Initialize a dictionary to store IoU for each angle and distance
    iou_per_distance_angle = {dist_value: {angle_value: [] for angle_value in angle} for dist_value in distance}
    iou_per_folder_angle = {folder_value: {angle_value: [] for angle_value in angle} for folder_value in folder} 
    IoU_values = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        statis = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img   = (gt).astype('uint8')
            pred_img = (pred > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))
    
            
          
        
                   
        
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
        
      
 
    final_accuracy_all = np.array(final_accuracy_all)
    max_iou_threshold = final_accuracy_all[np.argmax(final_accuracy_all[:, 4]), 0]
    print(f"Best IoU: {np.max(final_accuracy_all[:, 4]):.4f} at threshold {max_iou_threshold:.2f}")
    for pred, gt in zip(pred_list, gt_list):
            gt_img   = (gt).astype('uint8')
            pred_img = (pred > max_iou_threshold).astype('uint8')

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

    # Convert results to a NumPy array for easier plotting
    save_result_for_each_img(statis)
    # Convert 'statis' to a NumPy array before slicing
    statis = np.array(statis)

    p_acc_values = statis[:, 0]
    r_acc_values = statis[:, 1]
    f1_acc_values = statis[:, 2]
    IoU_values = statis[:, 3] 
    
    # Here you can log IoU based on angle and distance if available
    for i, img_name in enumerate(img_names):
        # Extract distance and angle from image name
        if distance != [] and angle != []:
            dist_index = next((dist for dist in distance if f'_{dist}_' in img_name), None)
            ang_index = next((ang for ang in angle if f'_{ang}_' in img_name), None)
            if 'gain' in img_name:
                folder_index = 'gain'
            elif 'gamma' in img_name:
               folder_index = 'gamma'
            else:
                folder_index = ''  # Default for the empty folder

            
                # Ensure the folder exists in the dictionary
            if dist_index and ang_index:
            # append iou for each each image to the corresponding distance and angle
                iou_per_folder_angle[folder_index][ang_index].append(statis[i][3])
                iou_per_distance_angle[dist_index][ang_index].append(statis[i][3])
            else:
                print(f"No matching distance/angle found for image {img_name}.")    
            if distance and angle: 
                plot_IoU_per_distance_angle(iou_per_distance_angle, distance, angle)
                plot_IoU_all_folders(iou_per_folder_angle, angle)    

    # Plot the metrics
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle(f'{model} metrics', fontsize=16)
    axs[0].plot(p_acc_values, color='y')
    axs[0].set_ylabel('Precision')
    axs[0].grid(True)

    axs[1].plot(r_acc_values,  color='r')
    axs[1].set_ylabel('Recall')
    axs[1].grid(True)

    axs[2].plot(f1_acc_values,  color='g')
    axs[2].set_ylabel('F1 Score')
    axs[2].grid(True)

    axs[3].plot(IoU_values, color='b')
    axs[3].set_xlabel('Images')
    axs[3].set_ylabel('IoU')
    axs[3].grid(True)

    

    # Save the figure
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H")
    plt.show()   
    fig.savefig(f"{current_datetime}_metrics.png")
    

    return final_accuracy_all

def plot_IoU_all_folders(iou_per_folder_angle, angle):
    fig, axs = plt.subplots(figsize=(15, 6))
    fig.suptitle(f'{model} IoU for Each Folder', fontsize=16)
    for folder_name in folder:
        plot_angles = []
        iou_values = []
        for ang in angle:
            if len(iou_per_folder_angle[folder_name][ang]) > 0:
                plot_angles.append(ang)
                iou_values.append(iou_per_folder_angle[folder_name][ang][0])  # Assuming one IoU value per angle
        if iou_values:
                plot_angles = np.asarray(plot_angles, dtype='float')
                axs.set_xticks(plot_angles)
                axs.plot(plot_angles, iou_values, marker='o', label=f'Distance 800cm {folder_name}')
                
    
    axs.set_xlabel('Angle')
    axs.set_ylabel('IoU')
    axs.legend()
    axs.grid(True)
    
    plt.show()
    current_time = datetime.now().strftime("%Y-%m-%d-%H")
    fig.savefig(f"{model}_IoU_All_Folders_{current_time}.png")

def plot_IoU_per_distance_angle(iou_per_distance_angle, distance, angle):
    fig, axs = plt.subplots(figsize=(15, 6))
    fig.suptitle(f'{model} IoU for each angle and distance', fontsize=16)
    #histogram for each distance and each angle
    # Initialize iou_values_per_angle as a list of empty lists
    iou_values_per_angle = [[] for _ in range(len(distance))]
    for dist in distance:
        iou_values_per_angle[distance.index(dist)].append(dist)  # Combine angle with its IoU values
        plot_angles = []
        iou_values = []
        for ang in angle:
            if len(iou_per_distance_angle[dist][ang]) > 0:
                plot_angles.append(ang)
                iou_values.append(iou_per_distance_angle[dist][ang][0])  # Assuming one IoU value per angle
                iou_values_per_angle[distance.index(dist)].append(iou_per_distance_angle[dist][ang][0])        
            else:
                iou_values_per_angle[distance.index(dist)].append(0)  
        # Plot IoU values for this distance
        if iou_values:
            plot_angles = np.asarray(plot_angles, dtype='float')
            axs.set_xticks(plot_angles)
            axs.plot(plot_angles, iou_values, marker='o', label=f'Distance {dist} mm')
           
            # axs2.set_yticks(iou_values)
        else:
            print(f"No IoU values found for distance {dist}")

        axs.set_ylim(0, 1)
        axs.set_xlabel('Angle in degrees')
        axs.set_ylabel(f'IoU')
        axs.legend()
        axs.grid(True)
    print(iou_values_per_angle)
    column_angle = angle.copy()
    column_angle.insert(0, 'Distance in mm')  # Add the Distance column
    print(plot_angles)
    df = pd.DataFrame(iou_values_per_angle, columns=column_angle)

            # Plot the grouped bar chart
    df.plot(
        x='Distance in mm',    # Column to use as x-axis
        kind='bar',      # Bar chart type
        stacked=False,   # Grouped bars
        title='IoU Values for Each Angle and Distance',
        figsize=(10, 6)  # Adjust figure size for readability
    )
    plt.xlabel("Distance in mm")
    plt.ylabel("IoU Value")
    plt.legend(title="Angles in degrees")  # Add legend with title
    plt.tight_layout()          # Adjust layout   
    plt.show()
    current_time = datetime.now().strftime("%Y-%m-%d-%H")
    fig.savefig(f"{model}_{current_time}_IoU.png")
    plt.savefig(f"{model}_{current_time}_histogram.png")


def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def save_results(input_list, output_path):
     # Convert input_list to a NumPy array for easy slicing
    input_array = np.array(input_list)
    
    # Find the best IoU and corresponding index
    best_iou = np.max(input_array[:, 4])
    best_iou_idx = np.argmax(input_array[:, 4])
    
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        fout.write("Threshold\tPrecision\tRecall\tF1\tIoU\n")
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)
        fout.write(f"Best IoU: {best_iou:.4f} at threshold {input_list[best_iou_idx, 0]:.2f}\n")

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
    fig2, axs = plt.subplots(1, 3, figsize=(15,5))
    fig, ax2 = plt.subplots(1, 1, figsize=(5,5))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].imshow(img/255.)

    axs[1].axis('off')
    axs[1].imshow(msk*255, cmap= 'gray')

    axs[2].axis('off')
    axs[2].imshow(mskp*255, cmap= 'gray')
    ax2.axis('off')
    ax2.imshow(mskp*255, cmap= 'gray')
    fig2.savefig(config['save_result'] + name + '.png')
    fig.savefig(config['save_result'] + name + '_pred.png')

config         = yaml.load(open('./config_crack.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# data_path = config['path_to_testdata']
# DIR_IMG  = os.path.join(data_path)
# DIR_MASK = os.path.join(data_path)
# img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
# mask_names = [path.name for path in Path(DIR_MASK).glob('*.png')]
# img_names= natsorted(img_names)
# mask_names=natsorted(mask_names)

distance = [250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]

angle = [10, 20, 45, 75, 90]
data_path = config['path_to_testdata']
DIR_IMG = [os.path.join(data_path, f'd_{d}/gamma') for d in distance] 
img_names = natsorted([path.name for img_dir in DIR_IMG for path in Path(img_dir).glob('*.jpg')])
DIR_MASK = [os.path.join(data_path, f'd_{d}/gamma') for d in distance]
mask_names = natsorted([path.name for mask_dir in DIR_MASK for path in Path(mask_dir).glob('*.png')])
# gain, gamma and d IoU values in one graph for 800

test_dataset = Crack_loader(img_dir=DIR_IMG, img_fnames=img_names, mask_dir=DIR_MASK, mask_fnames=mask_names)
test_loader  = DataLoader(test_dataset, batch_size = 1, shuffle= False)


print(f'test_dataset:{len(test_dataset)}')

#Net = TransMUNet(n_classes = number_classes)
Net = deepLab.deeplabv3plus_mobilenet(num_classes=number_classes, output_stride=8)
Net = Net.to(device)
Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])


# gt_list = [single_gt for _ in mask_names]
# gt_list.extend([single_gt] * len(angle))
pred_list = []
gt_list = []
save_samples = True # if save_samples=False, no samples will be saved.

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
        print('pred:', mskp.shape)
        end = time.time()
        times += (end - start)
        # print('print:', msk.numpy()[0,0])
        if itter < 238 and save_samples:
            save_sample(img_path, msk.numpy()[0, 0], mskp, name=img_names[itter])
            #save_sample(img_path, single_gt, mskp, name=str(itter+1))

        gt_list.append(msk.numpy()[0, 0])
        #gt_list.append(single_gt)
        pred_list.append(mskp)
    print('Running time of each images: %ss' % (times/len(pred_list)))

final_results = []
print(config['path_to_testdata'])
if config['path_to_testdata'] == "/data/Crack500/test" or config['path_to_testdata'] == "/data/DeepCrack/test" or config['path_to_testdata'] == "/data/combined/test":
    final_results = cal_prf_metrics(pred_list, gt_list, [], [], args.thresh_step, img_names=img_names)
else:
    final_results = cal_prf_metrics(pred_list, gt_list, distance, angle, args.thresh_step, img_names=img_names)

save_results(final_results, args.output)