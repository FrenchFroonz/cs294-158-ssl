import torch
import torch.nn as nn
import torch.nn.functional as F
from deepul_helper.resnet import resnet_v1
from deepul_helper.batch_norm import BatchNorm1d


class RelativePositionPrediction(nn.Module):
    """
    Pretext task: predict the relative position of a patch with respect to a center patch.
    Based on "Unsupervised Visual Representation Learning by Context Prediction" (Doersch et al., 2015)
    https://arxiv.org/pdf/1505.05192.pdf
    """
    metrics = ['Loss', 'Accuracy']
    metrics_fmt = [':.4e', ':6.2f']
    
    def __init__(self, dataset, n_classes):
        super().__init__()
        self.n_classes = n_classes
      
        if dataset == 'cifar10':
            self.encoder = resnet_v1((3, 8, 8), 18, 1, cifar_stem=True)
            self.latent_dim = 512  
        elif 'imagenet' in dataset:
            self.encoder = resnet_v1((3, 64, 64), 50, 1, cifar_stem=False)
            self.latent_dim = 2048
        
        self.position_classifier = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)  
        )
    
    def construct_classifier(self):
        return nn.Sequential(
            BatchNorm1d(self.latent_dim, center=False),
            nn.Linear(self.latent_dim, self.n_classes)
        )
    
    def forward(self, data):

        center_patches, neighbor_patches, position_labels = data
        
        z_center = self.encoder(center_patches)   
        z_neighbor = self.encoder(neighbor_patches)
        
        z_combined = torch.cat([z_center, z_neighbor], dim=1)
        
        logits = self.position_classifier(z_combined)
        
        loss = F.cross_entropy(logits, position_labels)
        
        _, predicted = logits.max(1)
        accuracy = (predicted == position_labels).float().mean() * 100
        
        return dict(Loss=loss, Accuracy=accuracy), z_center
    
    def encode(self, patches):
        return self.encoder(patches)
