import matplotlib.pyplot as plt
import numpy as np

# data

def plot_graph_mIoU(mIoUs, model_name):
    # data
    fig, axs = plt.subplots(figsize=(20, 10))
    fig.suptitle(f'{model_name} mIoU', fontsize=35)
    x = list(mIoUs.keys())
    y = list(mIoUs.values())
    axs.set_xticks(x)
    print("x", x)
    print("y", y)
    # plot
    axs.plot(x, y, marker='o', label=model_name)
    axs.set_xlabel('Angle (degrees)', fontsize=30)
    axs.set_ylabel('mIoU', fontsize=30)
     # Customize tick font sizes
    axs.tick_params(axis='x', labelsize=30)  # X-axis tick font size
    axs.tick_params(axis='y', labelsize=30)  # Y-axis tick font size
    
    # Add legend with custom font size
    axs.legend(fontsize=30)
    plt.show()
    fig.savefig(f'{model_name}_mIoU.png')