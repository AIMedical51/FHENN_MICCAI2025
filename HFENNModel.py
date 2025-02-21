import os
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F



class CustomResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained_path="./pretrained_models/resnet50-0676ba61.pth"):
        super(CustomResNet, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.load_state_dict(torch.load(pretrained_path))

        # Extract the number of features in the last layer
        self.features_dim = self.resnet.fc.in_features

        # Replace the classifier (fc layer) with a new one
        self.resnet.fc = nn.Identity()  # Remove the original fc layer
        self.classifier = nn.Linear(self.features_dim, num_classes)

    def forward(self, x):
        # Extract features from ResNet
        features = self.resnet(x)  # Shape: [batch_size, features_dim]

        # Compute logits using the classifier
        logits = self.classifier(features)  # Shape: [batch_size, num_classes]

        # Return logits, features, and classifier weights
        return logits, features, self.classifier.weight



# Dual Stage Model with Independent ResNets
class DualStageModel(nn.Module):
    def __init__(self, num_classes_stage1=8, num_classes_stage2=11, pretrained_path="../pretrained_models/resnet50-0676ba61.pth"):
        super(DualStageModel, self).__init__()
        
        # Stage 1 ResNet for three-class classification
        self.resnet_stage1 = models.resnet50(pretrained=False)
        self.resnet_stage1.load_state_dict(torch.load(pretrained_path))
        self.feature_dim_stage1 = self.resnet_stage1.fc.in_features
        self.resnet_stage1.fc = nn.Identity()
        self.stage1_classifier = nn.Linear(self.feature_dim_stage1, num_classes_stage1)

        # Stage 2 ResNet for fine-grained classification
        self.resnet_stage2 = models.resnet50(pretrained=False)
        self.resnet_stage2.load_state_dict(torch.load(pretrained_path))
        self.feature_dim_stage2 = self.resnet_stage2.fc.in_features
        self.resnet_stage2.fc = nn.Identity()
        self.stage2_classifier = nn.Linear(self.feature_dim_stage2 + self.feature_dim_stage1 + num_classes_stage1, num_classes_stage2)

    def forward(self, x):
        # Stage 1: Three-class classification
        features_stage1 = self.resnet_stage1(x)  # [batch_size, feature_dim_stage1]
        logits_stage1 = self.stage1_classifier(features_stage1)  # [batch_size, num_classes_stage1]
        
        # Stage 2: Fine-grained classification
        features_stage2 = self.resnet_stage2(x)  # [batch_size, feature_dim_stage2]
        combined_features = torch.cat([features_stage2, features_stage1, logits_stage1], dim=1)  # Combine features
        logits_stage2 = self.stage2_classifier(combined_features)  # [batch_size, num_classes_stage2]

        return logits_stage1, logits_stage2
    
    
class MoEClassifier(nn.Module):
    """
    Mixture of Experts (MoE) Classifier with custom evidence-based gating.
    """
    def __init__(self, input_dim, num_classes, num_experts=3):
        super(MoEClassifier, self).__init__()
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([nn.Linear(input_dim, num_classes) for _ in range(num_experts)])

    def forward(self, x):
        # Step 1: Compute logits for each expert
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, num_classes]
        
        # Step 2: Convert logits to evidence using ReLU
        expert_evidence = F.softplus(expert_logits)  # [batch_size, num_experts, num_classes]
        
        # Step 3: Compute total evidence for each expert
        total_evidence = torch.sum(expert_evidence, dim=2)  # [batch_size, num_experts]
        
        # Step 4: Compute uncertainty for each expert
        num_classes = expert_logits.size(2)  # C (number of classes)
        expert_uncertainty = num_classes / (total_evidence + 1e-8)  # [batch_size, num_experts]
        
        # Step 5: Compute weights (inverse of uncertainty)
        weights = 1 / expert_uncertainty  # [batch_size, num_experts]
        weights = weights / torch.sum(weights, dim=1, keepdim=True)  # Normalize weights [batch_size, num_experts]
        
        # Step 6: Compute fused evidence using weighted average
        fused_evidence = torch.sum(weights.unsqueeze(2) * expert_evidence, dim=1)  # [batch_size, num_classes]
        
        # Step 7: Compute mean logits
        mean_logits = torch.mean(expert_logits, dim=1)  # [batch_size, num_classes]
        
        return mean_logits,fused_evidence, expert_evidence, expert_uncertainty

class DualStageModelWithEvidenceFuse(nn.Module):
    def __init__(self, num_classes_stage1=3, num_classes_stage2=7, num_experts=3,pretrained_path="../pretrained_models/resnet50-0676ba61.pth"):
        super(DualStageModelWithEvidenceFuse, self).__init__()
        
        # Stage 1 ResNet for three-class classification
        self.resnet_stage1 = models.resnet50(pretrained=False)
        self.resnet_stage1.load_state_dict(torch.load(pretrained_path))
        self.feature_dim_stage1 = self.resnet_stage1.fc.in_features
        self.resnet_stage1.fc = nn.Identity()
        self.fuse_classifier1 = MoEClassifier(self.feature_dim_stage1, num_classes_stage1, num_experts)

        # Stage 2 ResNet for fine-grained classification
        self.resnet_stage2 = models.resnet50(pretrained=False)
        self.resnet_stage2.load_state_dict(torch.load(pretrained_path))
        self.feature_dim_stage2 = self.resnet_stage2.fc.in_features
        self.resnet_stage2.fc = nn.Identity()
        self.fuse_classifier2 = MoEClassifier(self.feature_dim_stage2 + self.feature_dim_stage1 + num_classes_stage1, num_classes_stage2, num_experts)
    def forward(self, x):
        # Stage 1: Three-class classification
        features_stage1 = self.resnet_stage1(x)  # [batch_size, feature_dim_stage1]
        logits1,fused_evidence1, expert_evidence1, expert_uncertainty1 = self.fuse_classifier1(features_stage1)  # [batch_size, num_classes_stage1]
        
        # Stage 2: Fine-grained classification
        features_stage2 = self.resnet_stage2(x)  # [batch_size, feature_dim_stage2]
        combined_features = torch.cat([features_stage2, features_stage1, logits1], dim=1)  # Combine features
        logits2,fused_evidence2, expert_evidence2, expert_uncertainty2 = self.fuse_classifier2(combined_features)  # [batch_size, num_classes_stage1]
        
        #moe_classifier = MoEClassifier(input_dim, num_classes, num_experts)
        #fused_evidence, expert_evidence, expert_uncertainty = moe_classifier(x)

        return logits1,fused_evidence1,logits2,fused_evidence2

class DualStageExtractFeatureModel(nn.Module):
    def __init__(self, num_classes_stage1=4, num_classes_stage2=7, pretrained_path="../pretrained_models/resnet50-0676ba61.pth"):
        super(DualStageExtractFeatureModel, self).__init__()
        
        # Stage 1 ResNet for three-class classification
        self.resnet_stage1 = models.resnet50(pretrained=False)
        self.resnet_stage1.load_state_dict(torch.load(pretrained_path))
        self.feature_dim_stage1 = self.resnet_stage1.fc.in_features
        self.resnet_stage1.fc = nn.Identity()
        self.stage1_classifier = nn.Linear(self.feature_dim_stage1, num_classes_stage1)

        # Stage 2 ResNet for fine-grained classification
        self.resnet_stage2 = models.resnet50(pretrained=False)
        self.resnet_stage2.load_state_dict(torch.load(pretrained_path))
        self.feature_dim_stage2 = self.resnet_stage2.fc.in_features
        self.resnet_stage2.fc = nn.Identity()
        self.stage2_classifier = nn.Linear(self.feature_dim_stage2 + self.feature_dim_stage1 + num_classes_stage1, num_classes_stage2)

    def forward(self, x):
        # Stage 1: Three-class classification
        features_stage1 = self.resnet_stage1(x)  # [batch_size, feature_dim_stage1]
        logits_stage1 = self.stage1_classifier(features_stage1)  # [batch_size, num_classes_stage1]
        
        # Stage 2: Fine-grained classification
        features_stage2 = self.resnet_stage2(x)  # [batch_size, feature_dim_stage2]
        combined_features = torch.cat([features_stage2, features_stage1, logits_stage1], dim=1)  # Combine features
        logits_stage2 = self.stage2_classifier(combined_features)  # [batch_size, num_classes_stage2]
        stage2_weights = self.stage2_classifier.weight  # [num_classes_stage2, input_dim]
        #stage2_bias = self.stage2_classifier.bias  # Optional: Bias terms, [num_classes_stage2]
        
        return combined_features,stage2_weights,logits_stage1, logits_stage2
