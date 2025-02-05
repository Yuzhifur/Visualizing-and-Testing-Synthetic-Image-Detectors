import time
import os
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import torch

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=5):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    since = time.time()
    
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train() 
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if (phase == 'train'):
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc)
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(
            best_model_params_path, weights_only=True))
    return model, train_losses, val_losses, train_accuracies, val_accuracies


def plot_training_curve(model_name, train_losses, val_losses, train_accuracies, val_accuracies):
    # Ensure inputs are converted to Python lists
    train_losses = [loss.cpu().item() if torch.is_tensor(loss)
                    else loss for loss in train_losses]
    val_losses = [loss.cpu().item() if torch.is_tensor(loss)
                  else loss for loss in val_losses]
    train_accuracies = [acc.cpu().item() if torch.is_tensor(
        acc) else acc for acc in train_accuracies]
    val_accuracies = [acc.cpu().item() if torch.is_tensor(acc)
                      else acc for acc in val_accuracies]

    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    loss_save_path = f"result\\{model_name+'_loss_curve.png'}"
    plt.savefig(loss_save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to {loss_save_path}")
    plt.close()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    acc_save_path = f"result\\{model_name+'_acc_curve.png'}"
    plt.savefig(acc_save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy curve saved to {acc_save_path}")
    plt.close()  # Close the figure