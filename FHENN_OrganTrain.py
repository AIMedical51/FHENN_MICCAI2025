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
from helpers import *
from losses import *
from datasets import *
from HFENNModel import *

def test_model(model, test_loader, device, coarselabelnum, finelabelnum):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        
        for inputs, coarse_labels,labels in test_loader:
            inputs= inputs.to(device)
            coarse_labels = coarse_labels.to(device).squeeze()
            labels = labels.to(device).squeeze()
       
            features,feature_weights,logits_stage1, logits_stage2 = model(inputs)
            
            evidence_feature=relu_evidence(features)#this features is the combined features
            alpha_evidence=evidence_feature+1
            total_S=torch.sum(alpha_evidence, dim=1, keepdim=True)
            hyper_belief=evidence_feature/total_S
            class_belief = torch.zeros(batchsize, finelabelnum, device=hyper_belief.device)
            positive_mask = (feature_weights > 0).float()  # [num_classes, 128]
            num_positive = positive_mask.sum(dim=0)  # [128]
            num_positive[num_positive == 0] = 1  
            contribution = hyper_belief / num_positive  # [batch_size, 128]
            contribution = contribution.unsqueeze(1) * positive_mask.unsqueeze(0)  # [batch_size, num_classes, 128]

            class_belief = contribution.sum(dim=2)  # [batch_size, num_fine_classes]
            _, preds = torch.max(class_belief, 1)
            probs=torch.nn.functional.softmax(class_belief, dim=1).cpu().numpy()  # Get probabilities            
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


def train_dual_stage_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=1, save_path="best_model.pth",coarselabelnum=8,finelabelnum=11,warmup=20):
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
            #image, coarse_label, fine_label
            for inputs, coarse_labels,labels in dataloader:
                inputs= inputs.to(device)
                coarse_labels = coarse_labels.to(device).squeeze()
                labels = labels.to(device).squeeze()
                fine_y=one_hot_embedding(labels,finelabelnum)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    features,feature_weights,logits_stage1, logits_stage2 = model(inputs)                 
                    coarse_y=one_hot_embedding(coarse_labels,coarselabelnum)
                    
                    ce_loss_stage1 = criterion_ce_stage1(logits_stage1, coarse_labels)
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
                       
                        evidence_feature=relu_evidence(features)#this features is the combined features
                        alpha_evidence=evidence_feature+1
                        
                        total_S=torch.sum(alpha_evidence, dim=1, keepdim=True)
                        hyper_belief=evidence_feature/total_S
                        class_belief = torch.zeros(batchsize, finelabelnum, device=hyper_belief.device)
                        positive_mask = (feature_weights > 0).float()  
                        num_positive = positive_mask.sum(dim=0) 
                        num_positive[num_positive == 0] = 1  
                        contribution = hyper_belief / num_positive  # [batch_size, 128]
                        contribution = contribution.unsqueeze(1) * positive_mask.unsqueeze(0)  # [batch_size, num_classes, 128]

                        class_belief = contribution.sum(dim=2)  # [batch_size, num_fine_classes]
                        _, preds = torch.max(class_belief, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            # Update learning rate
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

  
# Main function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load OrganSMNIST data
    file_path = "/home/lyn/.medmnist/organsmnist.npz"
    data = np.load(file_path, allow_pickle=True)
    batchsize=32
    train_images, train_labels = data['train_images'], data['train_labels']
    val_images, val_labels = data['val_images'], data['val_labels']
    test_images, test_labels = data['test_images'], data['test_labels']
    coarselabelnum=8
    finelabelnum=11
    def create_coarse_labels(labels):
        coarse_mapping = {
            0: 0, 1: 0,  
            2: 1, 3: 1,  
            4: 2,        
            5: 3,        
            6: 4,        
            7: 5,        
            8: 6, 9: 6, 
            10: 7        
        }
        return np.array([coarse_mapping[label.item()] for label in labels])
    train_coarse_labels = create_coarse_labels(train_labels)
    val_coarse_labels = create_coarse_labels(val_labels)
    test_coarse_labels = create_coarse_labels(test_labels)
    
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets and loaders
    train_dataset = OrganSMNISTDataset(train_images, train_coarse_labels, train_labels, transform=data_transforms)
    val_dataset = OrganSMNISTDataset(val_images, val_coarse_labels, val_labels, transform=data_transforms)
    test_dataset = OrganSMNISTDataset(test_images, test_coarse_labels, test_labels, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model
    # Initialize model
    model = DualStageExtractFeatureModel(num_classes_stage1=coarselabelnum, num_classes_stage2=finelabelnum, pretrained_path="../pretrained_models/resnet50-0676ba61.pth")
    model = model.to(device)


    # Define optimizer, scheduler, and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train model
    save_path = "./best_model.pth"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    model = train_dual_stage_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=100, save_path=save_path,coarselabelnum=coarselabelnum,finelabelnum=finelabelnum)
    print(f"Evaluating test model...")
    metrics = test_model(
            model, test_loader, device,coarselabelnum,finelabelnum
        )
    print(metrics) 