from models.customnet import CustomNet

def train(epoch, model, train_loader, criterion, optimizer):     #criterion (loss function
    model.train()  #SET TO TRAINING MODE
    running_loss = 0.0
    correct = 0    #correct counts correct predictions
    total = 0      #ALL PREDICTIONS



   # iterate over all batched by counting dataloader

    for batch_idx, (inputs, targets) in enumerate(train_loader):   #TARGET -> TRUE LABEL
        inputs, targets = inputs.cuda(), targets.cuda()

        # todo... <--- CORE OPTIMIZATION STEPS
        optimizer.zero_grad()      # 1. Zero the gradients
        outputs = model(inputs)    # 2. Forward pass
        loss = criterion(outputs, targets) # 3. Compute loss    #compare the results yoi get with true labels(target)
        loss.backward()            # 4. Backward pass (compute gradients)    #backprppgation
        optimizer.step()           # 5. Update weights

        running_loss += loss.item()     #ADD THE LOSS IF THIS BATCH TO ALL LSS COMPUTED
        _, predicted = outputs.max(1)    #MODELS FINAL CLASIFCATION THAT WANT TO FIND HIGHESTSCORE
        total += targets.size(0)         ##########ADD THE SIZE OF BATCH TO THE TOTAL
        correct += predicted.eq(targets).sum().item()   #It counts the number of correct matches

    train_loss = running_loss / len(train_loader)   #FINAL AVG LOSS
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')