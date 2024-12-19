import matplotlib.pyplot as plt
import numpy as np


# Plot the metrics
def plot_metrics(p_acc_values, r_acc_values, f1_acc_values, IoU_values, model):
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