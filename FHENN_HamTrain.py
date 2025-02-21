import os
import pandas as pd
from PIL import Image
from helpers import *
from losses import *
from sklearn.model_selection import KFold
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
from datasets import *
from HFENNModel import *

def test_model(model, test_loader, device, coarselabelnum, finelabelnum, fold=1):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
   
            # Forward pass
            features,feature_weights,logits_stage1, logits_stage2 = model(inputs)
            
            # Step 1: activate
            features = F.relu(features)  

            # Step 2: clone feature_weights 
            weights = feature_weights.clone().detach()  # [num_classes, feature_dim]

            # Step 3: compute new weights
            modified_weights = torch.zeros_like(weights)  # [num_classes, feature_dim]

            # Step 4: reassigin weight
            for class_idx in range(weights.size(0)): 
                for feature_idx in range(weights.size(1)):  
                    if weights[class_idx, feature_idx] > 0:
                        C = (weights[:, feature_idx] > 0).sum().item()
                        modified_weights[class_idx, feature_idx] = 1.0 / C  

            logits_stage2_new = torch.matmul(features, modified_weights.T)  # [batch_size, num_classes_stage2]

            # Step 6: caluate evidence_fuse
            evidence_fuse = F.relu(logits_stage2_new)            
            
            fused_alpha=(evidence_fuse+1)
            sum_fuse_alpha  = fused_alpha.sum(dim=1)
            
            _, preds = torch.max(fused_alpha, 1)
            probs=torch.nn.functional.softmax(fused_alpha, dim=1).cpu().numpy()  # Get probabilities            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs)

    # Compute metrics
    test_loss = 0
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    ece = compute_ece(all_labels_np, all_preds_np, np.array(all_probs), n_bins=15)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ECE Score: {ece:.4f}')

    return test_loss, accuracy, precision, recall, f1, ece


def train_dual_stage_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=1, save_path="best_model.pth",coarselabelnum=3,finelabelnum=7,warmup=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    criterion_ce_stage1 = nn.CrossEntropyLoss()  # Coarse
    criterion_ce_stage2 = nn.CrossEntropyLoss()  # Fine

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                fine_y=one_hot_embedding(labels,finelabelnum)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    features,feature_weights,logits_stage1, logits_stage2 = model(inputs)
                    labels_stage1 = torch.where((labels == 0) | (labels == 2) | (labels == 5) | (labels == 6), 
                            torch.tensor(0).to(device),  # Benign class
                            torch.where((labels == 1) | (labels == 3), 
                                        torch.tensor(1).to(device),  # Malignant class
                                        torch.tensor(2).to(device)))  # Transition class (label 4)
                    
                    #labels_stage1 = torch.where(labels < 4, labels, torch.tensor(3).to(device))  # Map to 4 classes
                    coarse_y=one_hot_embedding(labels_stage1,coarselabelnum)
                    
                    ce_loss_stage1 = criterion_ce_stage1(logits_stage1, labels_stage1)
                    ce_loss_stage2 = criterion_ce_stage2(logits_stage2, labels)

                    edl_loss_stage1 = edl_digamma_loss(logits_stage1, coarse_y, epoch, coarselabelnum, warmup, device)
                    edl_loss_stage2 = edl_digamma_loss(logits_stage2, fine_y,   epoch, finelabelnum, warmup, device)
                    ce_weight  = max(0.0, 1.0 - epoch / warmup)
                    edl_weight = min(1.0, epoch / warmup)
                    fused_loss_stage1 = ce_weight * ce_loss_stage1 + edl_weight * edl_loss_stage1
                    fused_loss_stage2 = ce_weight * ce_loss_stage2 + edl_weight * edl_loss_stage2

                    loss = 0.5*fused_loss_stage1+fused_loss_stage2
                    if(epoch<warmup):
                         _, preds = torch.max(logits_stage2, 1)
                    else:
                        features = F.relu(features)  

                        weights = feature_weights.clone().detach()  # [num_classes, feature_dim]

                        modified_weights = torch.zeros_like(weights)  # [num_classes, feature_dim]

                        for class_idx in range(weights.size(0)):  
                            for feature_idx in range(weights.size(1)):  
                                if weights[class_idx, feature_idx] > 0:
                                    C = (weights[:, feature_idx] > 0).sum().item()
                                    modified_weights[class_idx, feature_idx] = 1.0 / C  

                        logits_stage2_new = torch.matmul(features, modified_weights.T)  # [batch_size, num_classes_stage2]

                        evidence_fuse = F.relu(logits_stage2_new) 
                        fused_alpha=evidence_fuse+ 1  
                        _, preds = torch.max(fused_alpha, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path)
                print(f"New best model saved with Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts) 
    return model


# Main Code
if __name__ == "__main__":
    # Paths and transformations
    csv_file = '../dataset/HAM10000_metadata.csv'
    img_dir = ['../dataset/HAM10000_images_part_1/', '../dataset/HAM10000_images_part_2/']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read annotations
    annotations = pd.read_csv(csv_file)
# nv(0), bkl(2), vasc(5), df(6) => 0
# mel(1), bcc(3), akiec(4) => 1
# others => 2

    
    label_mapping = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
    annotations['label'] = annotations['dx'].map(label_mapping)

    # Data transformations
    data_transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Cross-validation setup
    k_folds = 5
    batchsize = 32
    learningrate = 1e-4
    coarselabelnum=4
    finelabelnum=7

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = {}
    results=[]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(annotations, annotations['label'])):
        print(f'Fold {fold + 1}/{k_folds}')
        print('-' * 10)

        train_annotations = annotations.iloc[train_idx].reset_index(drop=True)
        val_annotations = annotations.iloc[val_idx].reset_index(drop=True)

        train_dataset = HAM10000Dataset(train_annotations, img_dir, transform=data_transforms_train)
        val_dataset = HAM10000Dataset(val_annotations, img_dir, transform=data_transforms_val)

        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=4)

        # Initialize model
        
        model = DualStageExtractFeatureModel(num_classes_stage1=coarselabelnum, num_classes_stage2=finelabelnum, pretrained_path="../pretrained_models/resnet50-0676ba61.pth")
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learningrate, weight_decay=1e-2)
        scheduler = ExponentialLR(optimizer, gamma=0.9)     
        # Train model
        save_path = f"./FHENN_fold_{fold + 1}.pth"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        model = train_dual_stage_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=100, save_path=save_path,coarselabelnum=coarselabelnum,finelabelnum=finelabelnum)
        # Evaluate model
        print(f"Evaluating fold {fold + 1} model...")
        metrics = test_model(
            model, val_loader, device,coarselabelnum,finelabelnum, fold=fold + 1
        )


        results.append(metrics)
        
    results = np.array(results)
    mean_results = results.mean(axis=0)
    std_results = results.std(axis=0)
    print(f"HFENN method Average Metrics across {k_folds} folds:")
    print(f"Accuracy: {mean_results[1]:.4f} ± {std_results[1]:.4f}")
    print(f"Precision: {mean_results[2]:.4f} ± {std_results[2]:.4f}")
    print(f"Recall: {mean_results[3]:.4f} ± {std_results[3]:.4f}")
    print(f"F1: {mean_results[4]:.4f} ± {std_results[4]:.4f}")
    print(f"ECE: {mean_results[5]:.4f} ± {std_results[5]:.4f}")   