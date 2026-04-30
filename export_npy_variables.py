import numpy as np
from pathlib import Path

# Set your directory path here
directory = Path("./output/dated/20260430/test2_timingerror_swiftnondiv_sine_adhimex_ll_nt50dt2_0/")

# Loop over all .npy files
for file_path in sorted(directory.glob("*.npy")):
    name = file_path.stem  # filename without .npy extension
    data = np.load(file_path, allow_pickle=True)

    print(f"\n=== {name} ===")
    print(data)