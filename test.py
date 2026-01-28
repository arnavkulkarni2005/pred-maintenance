import os
import matplotlib.pyplot as plt
test_path = '/home/kulkarni/projects/pred_maintenance/plots/gallery/test_image.png'
plt.figure()
plt.plot([1, 2], [1, 2])
plt.savefig(test_path)
plt.close()

if os.path.exists(test_path):
    print("Success: Permissions are correct.")
else:
    print("Error: Directory is not writable or path is incorrect.")