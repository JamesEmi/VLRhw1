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
dataset = VOCDataset(split='test', size=(224, 224))  
indices = random.sample(range(len(dataset)), 1000)
subset = torch.utils.data.Subset(dataset, indices)

# Load the model
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = ResNet(20)
saved_model = torch.load('Q2Model_checkpoint.pth', map_location=device)  
model.load_state_dict(saved_model['model_states'])
model.eval()

features = []
labels = []
with torch.no_grad():
    for img, label, _ in subset:
        feature = model(img.unsqueeze(0))
        feature = torch.flatten(feature, 1)  
        features.append(feature.cpu().numpy().squeeze())
        labels.append(label.numpy())

features = np.vstack(features)
labels = np.vstack(labels)

tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)

colors = plt.cm.hsv(np.linspace(0, 1, len(VOCDataset.CLASS_NAMES))) #HSV for COLOR

mean_colors = np.dot(labels, colors)

mean_colors = mean_colors - np.min(mean_colors, axis=0)  # Min-Max Normalization
mean_colors = mean_colors / np.max(mean_colors, axis=0) 

plt.figure(figsize=(10, 10))
plt.scatter(
    features_2d[:, 0], features_2d[:, 1], 
    c=mean_colors, 
    s=50,              # Increase point size
    alpha=0.9         # Adjust opacity
)
plt.title("t-SNE Visualization of Image Features")

# Add a colorbar as legend
sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv)  # Using HSV for COLOR
sm.set_array([])  # Only needed for matplotlib < 3.1
cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, len(VOCDataset.CLASS_NAMES)))
cbar.set_ticklabels(VOCDataset.CLASS_NAMES)  # Add class names to colorbar

plt.savefig('TSNE_hsv.png')
