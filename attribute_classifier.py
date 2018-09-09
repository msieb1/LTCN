import argparse
import torch
import torch.nn as nn
from os.path import join
import os
import sys
import torchvision
import torchvision.transforms as transforms
from utils.dataloader import EmbeddingLoader
from tcn import DenseClassifier
from ipdb import set_trace

def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters 
    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 50
    batch_size = 8
    learning_rate = 0.001
    num_classes = 5
    n_views = 3
    # MNIST dataset 
    train_dataset = EmbeddingLoader(n_views, 
        emb_directory='/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/embeddings/pushing_rings/train', 
        label_directory='/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/embeddings/pushing_rings/labels/train/parsed'
                                               )

    val_dataset = EmbeddingLoader(n_views, 
        emb_directory='/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/embeddings/pushing_rings/valid', 
        label_directory='/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/embeddings/pushing_rings/labels/train/parsed'
                                               )
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)


    model = DenseClassifier(num_classes).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (embs, labels_1, labels_2) in enumerate(train_loader):  
            # Move tensors to the configured device
            embs = embs.to(device)
            labels_1 = labels_1.to(device).view(-1)
            labels_2 = labels_2.to(device).view(-1)

            # Forward pass
            outputs_1, outputs_2 = model(embs)

            loss_1 = criterion(outputs_1, labels_1)
            loss_2 = criterion(outputs_2, labels_2)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()


                    
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss active label: {:.4f},Loss passive label: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss_1.item(), loss_2.item()))

        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct_1 = 0
            total_1 = 0
            correct_2 = 0
            total_2 = 0
            for embs, labels_1, labels_2 in test_loader:
                embs = embs.to(device)
                labels_1 = labels_1.to(device).view(-1)
                labels_2 = labels_2.to(device).view(-1)

                outputs_1, outputs_2 = model(embs)

                _, predicted_1 = torch.max(outputs_1.data, 1)
                _, predicted_2 = torch.max(outputs_2.data, 1)

                total_1 += labels_1.size(0)
                correct_1 += (predicted_1 == labels_1).sum().item()
                total_2 += labels_2.size(0)
                correct_2 += (predicted_2 == labels_2).sum().item()
            print('Accuracy of active label network branch: {} %'.format(100 * correct_1 / total_1))
            print('Accuracy of passive label network branch: {} %'.format(100 * correct_2 / total_2))
            print("="*10)
    # Save the model checkpoint
    torch.save(model.state_dict(), join(args.model_folder, 'classfier-model.ckpt'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/pressing_orange_button')
    parser.add_argument('--experiments-folder', type=str, default='./experiments/pressing_orange_button')

    args = parser.parse_args()
    main(args)        
