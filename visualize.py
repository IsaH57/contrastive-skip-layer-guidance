import torch
import matplotlib.pyplot as plt
import numpy as np

PATH = 'SD3_similarity_tensor_full_dataset.pt'
MODEL= "SD3"




similarities = torch.load(PATH)
ones = torch.ones(similarities.size(0), 1)
similarities = torch.cat([ones, similarities], dim=1)


# Compute relative differences: change from layer i to i+1
relative_changes = similarities[:, 1:] - similarities[:, :-1]  # shape: [100, 19]

# Plotting
plt.figure(figsize=(8,5), dpi=200)

# Every individual relative change line in light gray
for i in range(relative_changes.shape[0]):
    plt.plot(range(19), relative_changes[i].cpu(), color='lightgray', linewidth=0.5)

# Average relative change as a thicker line using sampled color
mean_changes = relative_changes.mean(dim=0)
plt.plot(range(19), mean_changes.cpu(), color='#009456', linewidth=2.5, marker='o', alpha=1.)

# Plot styling
plt.axhline(y=0, color='black', linestyle='-', alpha=0.15)

plt.title(f"{MODEL}: Relative Change in Cosine Similarity per Layer")
plt.xlabel("Layer")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(19), [f"{i}" for i in range(19)])
plt.ylim(-0.2, 0.2)

plt.tight_layout()
plt.savefig(f"{MODEL}_relative_similarity_change.png", dpi=300)
plt.show()