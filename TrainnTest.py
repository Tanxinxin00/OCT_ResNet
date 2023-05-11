from tqdm import tqdm
import torch

def train(dataloader, model, criterion, optimizer, scaler, device):
    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    running_loss, running_accuracy = 0.0, 0.0
    
    for i, (x, y) in enumerate(dataloader):

        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        pred = model(x)

        accuracy = torch.mean((torch.argmax(pred, dim=1) == y).type(torch.float)).cpu()

        loss = criterion(pred, y)

        running_loss        += loss.item()
        running_accuracy    += accuracy
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()
        
        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(i+1)),
            accuracy="{:.04f}".format(running_accuracy/(i+1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()

        del x, y, pred, loss, accuracy
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_accuracy /= len(dataloader)
    batch_bar.close()

    return running_loss, running_accuracy

def test(dataloader,model, criterion,device):

    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Valid')

    running_loss, running_accuracy        = 0.0, 0.0
    
    for i, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)

        pred = model(x)

        accuracy = torch.mean((torch.argmax(pred, dim=1) == y).type(torch.float)).cpu()

        loss = criterion(pred, y)

        running_loss        += loss.item()
        running_accuracy    += accuracy
        
        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(i+1)),
            accuracy="{:.04f}".format(running_accuracy/(i+1)))
        batch_bar.update()

        del x, y, pred, loss, accuracy
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_accuracy /= len(dataloader)
    batch_bar.close()

    return running_loss, running_accuracy
