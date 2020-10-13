def train(args):
    """
    Note the following arguments expected with this implementation:
    
    1. optimizer: This should be a PyTorch optimizer function.
    2. loss: This should be the loss function as in above.
    3. learning_rate: The learning rate used for our optimizer.
    4. batch_size: The batch size during training.
    5. num_epochs: The number of training epochs.
    6. log_dir: The logging directory for the training dataset.

    """
    
    # Get paths for training and validation data
    cwd = os.getcwd()
    train_data_path = os.path.join(cwd, 'data', 'train')
    val_data_path = os.path.join(cwd, 'data', 'valid')
    
    # Create training and testing datasets
    training_dataset = SuperTuxDataset(train_data_path)
    val_dataset = SuperTuxDataset(val_data_path)
    
    # Create training and valiation DataLoaders out of datasets
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size)
    validate_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    
    # Create a CNN model
    model = CNNClassifier()
    
    # Create an optimizer
    opt = args.optimizer(model.parameters(), args.learning_rate)
    loss_function = args.loss
    
    # Lists for metrics
    avg_losses = {'train': [], 'valid': []}    # Keep track of losses
    avg_accuracy = {'train': [], 'valid': []}  # Keep track of accuracies
    
    # Create training loop
    for e in range(args.num_epochs):  # Iterate over dataset num_epochs times
        
        # Keep track of running loss
        running_loss_train = 0
        running_acc_train = 0
        i = 0
        
        # Set model to be in training mode before training
        model = model.train()
        
        # Training loop
        for img, label in training_dataloader:  # Iterate over batches in training set
            
            # Make predictions
            pred = predict(model, img)
            
            # Turn label into 
            
            # Compute accuaracy
            acc = accuracy(pred, label)

            # Calculate loss
            loss = loss_function.forward(pred, label, args.batch_size)

            # Zero the gradients to avoid gradient accumulation
            opt.zero_gradients()

            # Calculate the loss gradients
            loss.backward()

            # Take a step to update the neural network weights
            opt.step()

            # Update running loss and accuracy
            running_loss_train += loss.item()
            running_acc_train += acc

            # Print on some steps
            if i % 100 == 0:
                print("Running loss on epoch: {}".format(running_loss_train / (i+1)))

            i += 1
        
        # Add to accuracies and losses
        avg_losses['train'].append(running_loss_train / (i+1))
        avg_accuracy['train'].append(running_acc_train / (i+1))
        
        # Set model to be in evaluation mode before training
        model = model.eval()
        
        # Keep track of running loss
        running_loss_val = 0
        running_acc_val = 0
        j = 0
    
        # Evaluation loop
        for img, label in evaluation_dataloader:  # Iterate over images in validation set
            # Make predictions
            pred = predict(model, img)
            
            # Compute accuaracy
            acc = accuracy(pred, label)
            
            # Calculate loss
            loss = loss_function(pred, label)
            
            # Update losses and accuracies
            running_loss_val += loss.item()
            running_acc_val += acc
            
            j += 1
        
        
        # Add to accuracies and losses
        avg_losses['val'].append(running_loss_val / (i+1))
        avg_accuracy['val'].append(running_acc_val / (i+1))
        
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    save_model(model)
