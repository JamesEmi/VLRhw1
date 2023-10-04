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
dataset = VOCDataset(split='test', size=224, train=False)  # Adjust parameters as needed
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
colors = plt.cm.rainbow(np.linspace(0, 1, len(VOCDataset.CLASS_NAMES)))

# Compute the mean color for each sample
mean_colors = np.dot(labels, colors)

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=mean_colors, s=10)
plt.title("t-SNE Visualization of Image Features")

# Add a colorbar as legend
sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow)
sm.set_array(VOCDataset.CLASS_NAMES)
plt.colorbar(sm, ticks=range(len(VOCDataset.CLASS_NAMES)), boundaries=np.arange(len(VOCDataset.CLASS_NAMES)+1)-0.5)
plt.clim(-0.5, len(VOCDataset.CLASS_NAMES)-0.5)

# plt.show() 
plt.savefig('TSNE.png')
