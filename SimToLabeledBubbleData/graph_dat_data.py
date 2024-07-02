import os
import numpy as np
#from mayavi import mlab
import matplotlib.pyplot as plt
#from tvtk.util.ctf import ColorTransferFunction

# Read the binary .dat file
def read_dat_file(file_path):
    try:
        # Creates long string of integers for each data, not reshaped.
        data = np.fromfile(file_path, dtype=np.uint32)
    except Exception as e:
        print(f"Error reading or reshaping the file {file_path}: {e}")
        return None
    return data

# Ensure the output directory exists
output_dir = 'Images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each file in the VOFdata directory
input_dir = 'VOFdata'
for file_name in os.listdir(input_dir):
    if file_name.endswith('.dat'):
        file_path = os.path.join(input_dir, file_name)
        data = read_dat_file(file_path)
        if data is not None:
            nums = set()
            nums.update(data)
            nums = {x for x in nums if x >= 980000000}
            # print(nums)
            plt.hist(nums)
            plt.show()
            input("Press a key...")
            