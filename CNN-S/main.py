from dependencies import *
from models import *
from dataset_get import *

path = os.path.dirname(os.path.abspath(__file__)) 

class CNNcheck(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.CNNcell1 = CNNone(input_channels = 1, output_channels = 16)
        self.CNNcell2 = CNNone(input_channels = 16, output_channels = 32)
        self.CNNcell3 = CNNone(input_channels = 32, output_channels = 64)
        self.CNNcell4 = CNNone(input_channels = 64, output_channels = 64)
        self.CNNcell5 = CNNone(input_channels = 64, output_channels = 32)
        self.CNNcell6 = CNNone(input_channels = 32, output_channels = 32)
        #self.CNNcell7 = CNNone(input_channels = 32, output_channels = 16)
        self.CNNnetwork = nn.Sequential(self.CNNcell1, self.CNNcell2, 
                  self.CNNcell3,self.CNNcell4,self.CNNcell5,self.CNNcell6)
        self.fc1 = nn.Linear(in_features = 32*8*7, out_features = 2000)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features = 2000,out_features = 523*3)
    
    def forward(self, x):
        #x = x[:, None]
        x = self.CNNnetwork(x)
        x = x.view(-1, 32*8*7)
        x = self.dropout1(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def prepare_data(self):
        dataset = OCT_dataset(path, get_transform(train=True))
        dataset_test = OCT_dataset(path, get_transform(train=False))
        indices = torch.randperm(len(dataset)).tolist()
        size_of_test = int(len(dataset) * 0.15)
        size_of_main = len(dataset) - size_of_test
        
        dataset = torch.utils.data.Subset(dataset, indices[:-size_of_test])
        self.dataset_test = torch.utils.data.Subset(dataset_test, indices[-size_of_test:])
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset, [int(size_of_main*0.8),size_of_main - int(size_of_main*0.8)])

    def train_dataloader(self):
        oct_train = DataLoader(self.dataset_train, batch_size=64)
        return oct_train
    
    def val_dataloader(self):
        oct_val = DataLoader(self.dataset_val, batch_size=64)
        return oct_val
    
    def test_dataloader(self):
        oct_test = DataLoader(self.dataset_test, batch_size=64)
        return oct_test
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 1e-3)
        return optimizer

    def loss_funtion(self, input, target):
        return torch.sum((input - target) ** 2)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        input = self.forward(x)
        loss = self.loss_funtion(input, y)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        input = self.forward(x)
        loss = self.loss_funtion(input, y)
        return {'val_loss': loss}

    
    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        input = self.forward(x)
        return {'test_loss': self.loss_funtion(input, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}


if __name__ == "__main__":
    cnn_oct = CNNcheck()
# most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(gpus=1,profiler=True)    
    trainer.fit(cnn_oct)  
    