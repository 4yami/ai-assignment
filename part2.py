import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Dataset path
train_dir = "final_data/train"
classes = os.listdir(train_dir)

plt.figure(figsize=(12, 8))
img_count = 1

for class_name in classes:
    class_path = os.path.join(train_dir, class_name)
    images = random.sample(os.listdir(class_path), 3)
    
    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        # Load image using tf.keras.utils.load_img
        img = tf.keras.utils.load_img(img_path, color_mode="grayscale")
        # Convert to array
        img_array = tf.keras.utils.img_to_array(img)
        
        plt.subplot(len(classes), 3, img_count)
        plt.imshow(tf.squeeze(img_array), cmap="gray")
        plt.title(class_name)
        plt.axis("off")
        img_count += 1

plt.tight_layout()
plt.show()
