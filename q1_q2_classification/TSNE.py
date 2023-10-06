import torch
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from voc_dataset import VOCDataset
import random
from train_q2 import ResNet

# Load the data
# Assuming VOCDataset is a Dataset class for PASCAL VOC
dataset = VOCDataset(split='test', size=224)  # Adjust parameters as needed
indices = random.sample(range(len(dataset)), 1000)
subset = torch.utils.data.Subset(dataset, indices)
dataloader = torch.utils.data.DataLoader(subset, batch_size=1000)

# Extract features
# Assume model is a pretrained CNN (like ResNet-18) with modified output features for PASCAL VOC

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

model = ResNet(20)
saved_model = torch.load('Q2Model_checkpoint.pth', map_location=device)  # Load your model here
model.load_state_dict(saved_model['model_states'])
model.eval()

# Here, we'll assume the feature extractor is the model without its final classification layer
# Depending on your model architecture, you might extract features from a different layer
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))

features, labels = None, None
for inputs, targets, _ in dataloader:
    with torch.no_grad():
        outputs = feature_extractor(inputs)
        outputs = outputs.view(outputs.size(0), -1)  # Flatten the features
    features = outputs.cpu().numpy() if features is None else np.vstack((features, outputs.cpu().numpy()))
    labels = targets.cpu().numpy() if labels is None else np.vstack((labels, targets.cpu().numpy()))

# Compute t-SNE
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

# Create a color for each class
colors = plt.cm.tab20(np.linspace(0, 1, len(VOCDataset.CLASS_NAMES)))

# Compute the mean color for each sample
mean_colors = np.dot(labels, colors)

# Normalize to [0, 1]A
mean_colors = mean_colors - np.min(mean_colors, axis=0)  # Min-Max Normalization
mean_colors = mean_colors / np.max(mean_colors, axis=0) 

plt.figure(figsize=(10, 10))
plt.scatter(
    features_2d[:, 0], features_2d[:, 1], 
    c=mean_colors, 
    s=50,              # Increase point size
    # edgecolor='k',     # Add edge color
    # linewidth=0.5,     # Adjust edge width
    # marker='o',        # Adjust marker style
    alpha=0.9         # Adjust opacity
)
plt.title("t-SNE Visualization of Image Features")


# Add a colorbar as legend
sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20)  # Using 'tab20' colormap for colorbar
sm.set_array([])  # Only needed for matplotlib < 3.1
cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, len(VOCDataset.CLASS_NAMES)))
cbar.set_ticklabels(VOCDataset.CLASS_NAMES)  # Add class names to colorbar

# plt.show() 
plt.savefig('TSNE_trial_tab20.png')
