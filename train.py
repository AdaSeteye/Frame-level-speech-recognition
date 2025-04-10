from imports import device, torch
from architecture import Network
import config
from data_loader import train_data, val_loader, train_loader
from utils import train, eval


INPUT_SIZE  = (2*config['context'] + 1) * 28 
model       = Network(INPUT_SIZE, len(train_data.phonemes)).to(device).cuda()


criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


scaler = torch.amp.GradScaler('cuda', enabled=True)


torch.cuda.empty_cache()
gc.collect()


best_val_acc = 0.0

for epoch in range(config['epochs']):

    print("\nEpoch {}/{}".format(epoch+1,  config['epochs']))

    curr_lr                 = float(optimizer.param_groups[0]['lr'])
    train_loss, train_acc   = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc       = eval(model, val_loader)

    print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc*100, train_loss, curr_lr))
    print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc*100, val_loss))

    
   
    scheduler.step(val_acc)

    
