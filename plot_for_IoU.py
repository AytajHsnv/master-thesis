import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime


def plot_IoU_per_distance_angle(iou_per_distance_angle, distance, angle, model):
    fig, axs = plt.subplots(figsize=(15, 6))
    fig.suptitle(f'{model} IoU for each angle and pipe diameter', fontsize=16)
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
            axs.plot(plot_angles, iou_values, marker='o', label=f'Pipe diameter {dist} mm')
           
            # axs2.set_yticks(iou_values)
        else:
            print(f"No IoU values found for distance {dist}")

        axs.set_ylim(0, 1)
        axs.set_xlabel('Angle in degrees')
        axs.set_ylabel(f'IoU')
        axs.legend()
        axs.grid(True)
    column_angle = angle.copy()
    column_angle.insert(0, 'Pipe diameter in mm')  # Add the Distance column
    df = pd.DataFrame(iou_values_per_angle, columns=column_angle)

            # Plot the grouped bar chart
    df.plot(
        x='Pipe diameter in mm',    # Column to use as x-axis
        kind='bar',      # Bar chart type
        stacked=False,   # Grouped bars
        title='IoU Values for Each Angle and Distance',
        figsize=(30, 10)  # Adjust figure size for readability
    )
    plt.title("IoU Values for Each Angle and Distance", fontsize=50)  # Set a higher font size for the title
    plt.xlabel("Pipe diameter in mm", fontsize=45)
    plt.ylabel("IoU Value", fontsize=45)
    plt.legend(title="Angles in degrees",  title_fontsize=45, fontsize=45)  # Add legend with title
    plt.xticks(fontsize=45)  # Adjust font size for readability
    plt.yticks(fontsize=45)  # Adjust font size for readability
    plt.tight_layout()          # Adjust layout   
    plt.show()
    current_time = datetime.now().strftime("%Y-%m-%d-%H")
    fig.savefig(f"{model}_{current_time}_IoU.png")
    plt.savefig(f"{model}_{current_time}_histogram.png")
