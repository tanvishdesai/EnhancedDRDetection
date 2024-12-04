import numpy as np

# Load resampled data
data = np.load(r"C:\Users\DELL\Downloads\resampled_data.npz")

X_resampled = data['images']  # Shape: (num_samples, 3, 224, 224)
y_resampled = data['labels']

print("Number of samples in resampled dataset:", X_resampled.shape[0])


import numpy as np
from collections import Counter

# data = np.load('resampled_data.npz')
y_resampled = data['labels']

class_counts = Counter(y_resampled)
print("Class distribution after resampling:", class_counts)


import matplotlib.pyplot as plt

# # Visualize some images
# for i in range(5):  # Show 5 synthetic samples
#     plt.imshow(X_resampled[i].transpose(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
#     plt.title(f"Class: {y_resampled[i]}")
#     plt.show()
