import torch
import matplotlib.pyplot as plt
import numpy as np

path = "SD3_similarity_tensor_full_dataset.pt"
MODEL= "SD3"

def plot_relative_similarity_change(similarity_tensor_path):
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    # load similarity tensor
    similarity_tensor = torch.load(similarity_tensor_path)

    similarity_array = similarity_tensor.numpy()

    # calculate relative changes
    relative_changes = np.zeros_like(similarity_array)
    for i in range(similarity_array.shape[0]):
        for j in range(1, similarity_array.shape[1]):
            prev_val = similarity_array[i, j - 1]
            if prev_val != 0:
                relative_changes[i, j] = (similarity_array[i, j] - prev_val) / prev_val
            else:
                relative_changes[i, j] = 0

    # first column is always 0 (no previous layer)
    relative_changes[:, 0] = 0

    plt.figure(figsize=(12, 8))

    # every individual relative change line in light gray
    for i in range(relative_changes.shape[0]):
        plt.plot(range(1, relative_changes.shape[1]), relative_changes[i, 1:],
                 color='lightgray', linewidth=0.5)

    # average relative change as a thicker red line
    mean_changes = relative_changes.mean(axis=0)
    plt.plot(range(1, len(mean_changes)), mean_changes[1:],
             color='red', linewidth=2.5, marker='o')

    plt.title(f"{MODEL}: Relative Change in Cosine Similarity per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Relative change in Cosine Similarity from previous layer")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(1, similarity_array.shape[1]),
               [f"{i}" for i in range(1, similarity_array.shape[1])])


    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{MODEL}_relative_similarity_change.png", dpi=300)
    plt.show()

    return relative_changes, mean_changes

relative_changes, mean_changes = plot_relative_similarity_change(path)