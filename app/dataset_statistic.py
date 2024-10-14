import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

from app.model import generate_data_paths
from config import DATA_DIR, RESULT_PATH

def generate_data_statistics(data_dir):
    save_path =os.path.join(RESULT_PATH, 'class_distribution_pie_chart.png')
    # Generate file paths and labels
    filepaths, labels = generate_data_paths(data_dir)
    df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})

    # Dataset size and class distribution
    num_images = df.shape[0]
    num_classes = df['labels'].nunique()
    class_distribution = df['labels'].value_counts()

    # Initialize image dimension variables
    total_width, total_height, total_filesize = 0, 0, 0
    count = 0

    # Iterate through images to calculate average dimensions and file sizes
    for filepath in df['filepaths']:
        with Image.open(filepath) as img:
            width, height = img.size
            total_width += width
            total_height += height
            total_filesize += os.path.getsize(filepath)
            count += 1

    # Calculate average dimensions and file size
    avg_width = total_width / count if count > 0 else 0
    avg_height = total_height / count if count > 0 else 0
    avg_filesize_kb = (total_filesize / count / 1024) if count > 0 else 0

    # Print statistics
    print(f"Total Images: {num_images}")
    print(f"Number of Classes: {num_classes}")
    print(f"Class Distribution:\n{class_distribution}")
    print(f"Average Image Width: {avg_width:.2f} pixels")
    print(f"Average Image Height: {avg_height:.2f} pixels")
    print(f"Average File Size: {avg_filesize_kb:.2f} KB")

    # Generate pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=90,
            colors=plt.cm.Paired.colors)
    plt.title('Class Distribution')
    plt.savefig(save_path)
    print(f"Pie chart saved at {save_path}")

    # Return statistics in a dictionary format
    stats = {
        'total_images': num_images,
        'num_classes': num_classes,
        'class_distribution': class_distribution,
        'average_width': avg_width,
        'average_height': avg_height,
        'average_filesize_kb': avg_filesize_kb
    }
    return stats
# Example usage
stats = generate_data_statistics(DATA_DIR)
