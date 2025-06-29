import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def visualize_dataset(df,features=['feature_0', 'feature_1'], classes = 3, title= 'None'):
    palette = sns.color_palette('deep', classes)
    class_color = {i: palette[i] for i in range(classes)}
    
    y_min, y_max = df['feature_1'].min(), df['feature_1'].max()
    x_min, x_max = df['feature_0'].min(), df['feature_0'].max()
    radius = 0.05  # adjust radius to taste (in data units)

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        x0, x1 = row['feature_0'], row['feature_1']
        y = np.array(row['target'])
        

        mask = (y > 0.5)
        count = mask.sum()

        angle_span = 360.0 / count
        start_angle = 0.0
        for idx, active in enumerate(mask):
            if not active:
                continue
            wedge = mpatches.Wedge(
                center=(x0, x1),
                r=radius,
                theta1=start_angle,
                theta2=start_angle + angle_span,
                facecolor = class_color[idx],
                edgecolor='k',
                linewidth=0.5
            )
            ax.add_patch(wedge)
            start_angle += angle_span

    ax.set_xlim(x_min - 2*radius, x_max + 2*radius)
    ax.set_ylim(y_min - 2*radius, y_max + 2*radius)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_title(title)

    # Legend patches
    legend_patches = [mpatches.Patch(color=class_color[i], label=f'Class {i}') for i in class_color]
    ax.legend(handles=legend_patches, title='Classes')

    plt.tight_layout()
    plt.show()