
import torch
#from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score



def train_epoch(model, criterion, optimizer, dataloader, device, epoch, log_interval, auroc):
    model.train()
    losses = []
    all_label = []
    all_pred = []

    for batch_idx, batch in enumerate(dataloader):
      
        # get the inputs and labels 
        inputs, labels = batch['seq'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]

        # compute the loss
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels)
        all_pred.extend(prediction)
        acc = accuracy_score(labels.cpu().data.numpy(), prediction.cpu().data.numpy())

        # backward & optimize
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            #print(f"outputs: {outputs}, prediction: {prediction}")
            print("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), acc*100))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_auc = auroc(all_label.squeeze().cpu().data.squeeze(), all_pred.cpu().data.squeeze())

    # Log
    # writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    # writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    print("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | AUC {:.2f}%".format(epoch+1, training_loss, training_acc*100, training_auc*100))



def val_epoch(model, criterion, dataloader, device, epoch, auroc):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:

            # get the inputs and labels
            inputs, labels = batch['seq'].to(device), batch['label'].to(device)

            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]

            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())

            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)

    # Compute the average loss & accuracy
    val_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    val_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    val_auc = auroc(all_label.squeeze().cpu().data.squeeze(), all_pred.cpu().data.squeeze())
    # Log
    # writer.add_scalars('Loss', {'val': val_loss}, epoch+1)
    # writer.add_scalars('Accuracy', {'val': val_acc}, epoch+1)
    print("Average Validation Loss: {:.6f} | Acc: {:.2f}% | AUC {:.2f}%".format(val_loss, val_acc*100, val_auc*100))