import os
import pandas as pd
from PIL import Image
from helpers import *
from losses import *
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F


# Custom Dataset
class HAM10000Dataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['image_id'] + '.jpg'
        label = self.annotations.iloc[idx]['label']

        image = None
        for directory in self.img_dir:
            img_path = os.path.join(directory, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                break

        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found in specified directories.")

        if self.transform:
            image = self.transform(image)
        return image, label, img_name


# Dual Stage Model with Independent ResNets
class DualStageExtractFeatureModel(nn.Module):
    def __init__(self, num_classes_stage1=4, num_classes_stage2=7, pretrained_path="../pretrained_models/resnet50-0676ba61.pth"):
        super(DualStageExtractFeatureModel, self).__init__()
        
        # Stage 1 ResNet
        self.resnet_stage1 = models.resnet50(pretrained=False)
        self.resnet_stage1.load_state_dict(torch.load(pretrained_path))
        self.feature_dim_stage1 = self.resnet_stage1.fc.in_features
        self.resnet_stage1.fc = nn.Identity()
        self.stage1_classifier = nn.Linear(self.feature_dim_stage1, num_classes_stage1)

        # Stage 2 ResNet
        self.resnet_stage2 = models.resnet50(pretrained=False)
        self.resnet_stage2.load_state_dict(torch.load(pretrained_path))
        self.feature_dim_stage2 = self.resnet_stage2.fc.in_features
        self.resnet_stage2.fc = nn.Identity()
        self.stage2_classifier = nn.Linear(
            self.feature_dim_stage2 + self.feature_dim_stage1 + num_classes_stage1,
            num_classes_stage2
        )

    def forward(self, x):
        # Stage 1
        features_stage1 = self.resnet_stage1(x)  
        logits_stage1 = self.stage1_classifier(features_stage1)

        # Stage 2
        features_stage2 = self.resnet_stage2(x)
        combined_features = torch.cat([features_stage2, features_stage1, logits_stage1], dim=1)
        logits_stage2 = self.stage2_classifier(combined_features)

        stage2_weights = self.stage2_classifier.weight
        return combined_features, stage2_weights, logits_stage1, logits_stage2


def test_model_with_details(model, test_loader, device, output_path="test_results.csv", scale=0.01):
    model.eval()
    running_loss = 0.0
    results = []

    with torch.no_grad():
        for inputs, labels, img_names in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            features, feature_weights, logits_stage1, logits_stage2 = model(inputs)

            # Step 1: actiavte
            features = F.relu(features)

            # Step 2: clone feature_weights
            weights = feature_weights.clone().detach()  # [num_classes_stage2, feature_dim]

            # Step 3: compute new weights
            modified_weights = torch.zeros_like(weights)  # [num_classes_stage2, feature_dim]

            # Step 4: assign
            for class_idx in range(weights.size(0)):  
                for feature_idx in range(weights.size(1)):
                    if weights[class_idx, feature_idx] > 0:
                        
                        C = (weights[:, feature_idx] > 0).sum().item()
                        modified_weights[class_idx, feature_idx] = 1.0 / C

            # Step 5: projetion
            logits_stage2_new = torch.matmul(features, modified_weights.T)  # [batch_size, num_classes_stage2]

            # Step 6: compute evidence_fuse
            evidence_fuse = F.relu(logits_stage2_new)
            fused_alpha = (evidence_fuse + 1) * scale

            # prediction 
            _, preds = torch.max(fused_alpha, 1)
            sum_fuse_alpha = fused_alpha.sum(dim=1)
            
            probs = (fused_alpha / sum_fuse_alpha.unsqueeze(1)).cpu().numpy()

            
            for i in range(len(labels)):
                img_name = img_names[i]
                p_evidential = probs[i][preds[i]].item()  
                correct = (preds[i].item() == labels[i].item())
                correct_class_p = probs[i][labels[i]].item() if not correct else None

                results.append({
                    "img_name": img_name,
                    "p_evidential": p_evidential,
                    "prediction": preds[i].item(),
                    "true_label": labels[i].item(),
                    "correct": "correct" if correct else "wrong",
                    "correct_class_p": correct_class_p
                })

    test_loss = running_loss / len(test_loader.dataset) if len(test_loader.dataset) else 0.0
    accuracy = sum(1 for r in results if r["correct"] == "correct") / len(results)

    # save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return test_loss, accuracy, results_df


# ============ Main Code ============= #
if __name__ == "__main__":
    
    csv_file = '../dataset/HAM10000_metadata.csv'
    img_dir  = ['../dataset/HAM10000_images_part_1/', '../dataset/HAM10000_images_part_2/']
    train_csv = "train.csv"
    val_csv   = "val.csv"
    test_csv  = "test.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    annotations = pd.read_csv(csv_file)
    label_mapping = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
    annotations['label'] = annotations['dx'].map(label_mapping)
    
    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        print("Splits already exist. Loading from train.csv, val.csv, test.csv...")
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        test_df  = pd.read_csv(test_csv)
    else:
        print("Splits not found. Performing new train/val/test split (70/15/15) ...")
        train_df, testval_df = train_test_split(
            annotations,
            test_size=0.30,
            random_state=42,
            stratify=annotations['label']
        )
        val_df, test_df = train_test_split(
            testval_df,
            test_size=0.5,
            random_state=42,
            stratify=testval_df['label']
        )
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)

    print(f"Train set: {len(train_df)}, Val set: {len(val_df)}, Test set: {len(test_df)}")


    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # DataLoader
    train_dataset = HAM10000Dataset(train_df, img_dir, transform=data_transforms_train)
    val_dataset   = HAM10000Dataset(val_df,   img_dir, transform=data_transforms_val)
    test_dataset  = HAM10000Dataset(test_df,  img_dir, transform=data_transforms_val)

    batchsize = 32
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batchsize, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batchsize, shuffle=False, num_workers=4)

    
    coarselabelnum = 3
    finelabelnum   = 7
    model = DualStageExtractFeatureModel(
        num_classes_stage1=coarselabelnum,
        num_classes_stage2=finelabelnum,
        pretrained_path="../pretrained_models/resnet50-0676ba61.pth"
    ).to(device)

   
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    

    model.load_state_dict(torch.load("../saved_fixed_bm_split_hyperedl/best_model.pth"))
    model = model.to(device)

    print("\nEvaluating on test set...")
    output_path = "./results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    test_loss, accuracy, results_df = test_model_with_details(
        model, test_loader, device, output_path
    )

    unique_labels = sorted(results_df["true_label"].unique())
    print("\nPer-class Accuracy:")
    for lbl in unique_labels:
        subset_df = results_df[results_df["true_label"] == lbl]
        total_count = len(subset_df)
        correct_count = sum(subset_df["correct"] == "correct")
        acc_lbl = correct_count / total_count if total_count > 0 else 0.0
        print(f"  Label {lbl}: count={total_count}, acc={acc_lbl:.4f}")