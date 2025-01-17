from lung_cancer_detection.tf_dataset_loader import load_hf_datasets 
import numpy as np 

test_dataset = load_hf_datasets("data/metadata/test.csv")

# Define the mapping for string labels to numbers
label_mapping = {"aca": 0, "normal": 1, "scc": 2}

# Extract labels and map them to numbers
labels = [label_mapping[example["labels"]] for example in test_dataset]

# Convert to a NumPy array if needed
labels = np.array(labels)

print(labels)