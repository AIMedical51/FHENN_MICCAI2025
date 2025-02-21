import os
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import seaborn as sns
import torch.nn as nn
import copy
import pandas as pd
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


def test_model(model, test_loader, device, coarselabelnum, finelabelnum,cm_save_path='result.png'):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    uncertainty_collect=[]
    correct_uncertainty_collect=[]
    incorrect_uncertainty_collect=[]
    with torch.no_grad():
        
        for inputs, coarse_labels,labels in test_loader:
            inputs= inputs.to(device)
            std=0.2#0.025
            noise = torch.randn_like(inputs) * std
            
            noisy_inputs = inputs+noise
            coarse_labels = coarse_labels.to(device).squeeze()
            labels = labels.to(device).squeeze()
       
            features,feature_weights,logits_stage1, logits_stage2 = model(noisy_inputs)
            
            evidence_feature=relu_evidence(features)
            alpha_evidence=evidence_feature+1
            total_S=torch.sum(alpha_evidence, dim=1, keepdim=True)
            alpha_logits=relu_evidence(logits_stage2)+1
            uncertainty_total_S=torch.sum(alpha_logits, dim=1, keepdim=True)
            hyper_belief=evidence_feature/total_S
            class_belief = torch.zeros(batchsize, finelabelnum, device=hyper_belief.device)
            positive_mask = (feature_weights > 0).float()  # [num_classes, 128]
            num_positive = positive_mask.sum(dim=0)  # [128]
            num_positive[num_positive == 0] = 1  
            contribution = hyper_belief / num_positive  # [batch_size, 128]
            contribution = contribution.unsqueeze(1) * positive_mask.unsqueeze(0)  # [batch_size, num_classes, 128]

            class_belief = contribution.sum(dim=2)  # [batch_size, num_fine_classes]
            _, preds = torch.max(class_belief, 1)
            uncertainty=finelabelnum/(uncertainty_total_S)
            
            correct_indices = (preds == labels).nonzero(as_tuple=True)[0]
            incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
            
            correct_uncertainty = uncertainty[correct_indices]
            correct_uncertainty_collect.extend(correct_uncertainty.cpu().numpy())
            incorrect_uncertainty = uncertainty[incorrect_indices]
            incorrect_uncertainty_collect.extend(incorrect_uncertainty.cpu().numpy())
            print("uncertainty=",uncertainty)
            probs=torch.nn.functional.softmax(class_belief, dim=1).cpu().numpy()  # Get probabilities            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs)
        
        acc, sen, spe, f1 = compute_metrics(all_labels, all_preds, finelabelnum)
    correct_uncertainty_values = np.array(correct_uncertainty_collect)
    incorrect_uncertainty_values=np.array(incorrect_uncertainty_collect)
    
    step_values = np.arange(0, 1.000125, 0.000125)  


    min_length = min(len(correct_uncertainty_values), len(step_values))  
    correct_uncertainty_values = correct_uncertainty_values[:min_length]
    step_values = step_values[:min_length]
    correct_uncertainty_values = correct_uncertainty_values.flatten()
    step_values = step_values.flatten()
    index_values = np.arange(1, len(correct_uncertainty_values) + 1)

    df = pd.DataFrame({
        "Index": index_values,
        "Uncertainty": correct_uncertainty_values,
        "Step": step_values
    })
    df.to_csv(f"index_correct_nosiy_{std}.csv", index=False)


    step_values = np.arange(0, 1.000125, 0.000125) 
    min_length = min(len(incorrect_uncertainty_values), len(step_values))  
    incorrect_uncertainty_values = incorrect_uncertainty_values[:min_length]
    step_values = step_values[:min_length]
    incorrect_uncertainty_values = incorrect_uncertainty_values.flatten()
    step_values = step_values.flatten()
    index_values = np.arange(1, len(incorrect_uncertainty_values) + 1)
    df = pd.DataFrame({
        "Index": index_values,
        "Uncertainty": incorrect_uncertainty_values,
        "Step": step_values
    })
    df.to_csv(f"index_incorrect_nosiy_{std}.csv", index=False)
   

   
    test_loss = 0
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    class_acc, overall_acc = compute_metrics_per_class(all_labels_np, all_preds_np, finelabelnum)
    # Save results to file
    
    print(f"{overall_acc:.4f}\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\n")
    print("Per-class ACC:\n")
    for i, accc in enumerate(class_acc):
        print(f"Class {i}: {accc:.4f}\n")
        
    unique, counts = np.unique(all_labels_np, return_counts=True)
    
    print("Class distribution in validation set:\n")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count}\n")
    
    
    ece = compute_ece(all_labels_np, all_preds_np, np.array(all_probs), n_bins=15)
    # Generate confusion matrix
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=list(range(finelabelnum)))
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(finelabelnum)])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    plt.title("Confusion Matrix for Fine Classes")
  
    plt.savefig(cm_save_path, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_save_path}")
   
    #print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {acc:.4f}')
    print(f'SEN: {sen:.4f}')
    print(f'spe: {spe:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ECE Score: {ece:.4f}')
    
    return acc,sen,spe,f1,ece,class_acc



  
# Main function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load OrganSMNIST data
    file_path = "/.medmnist/organsmnist.npz"
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
    model.load_state_dict(torch.load(save_path))
    print(f"Evaluating test model...")
    metrics = test_model(
            model, test_loader, device,coarselabelnum,finelabelnum
        )
    print(metrics)
    