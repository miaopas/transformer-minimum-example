import argparse
from re import A
import torch
import numpy as np
from models.vit_lit import VitModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import ml_collections
from models.modeling import  CONFIGS
from utils.data_utils import get_loader
from math import floor

def get_default_args():
    args = ml_collections.ConfigDict()
    args.model_type = None
    args.num_classes = None
    args.dataset = None
    args.name = None
    args.num_steps = None
    args.device = None
    args.gradient_accumulation_steps = None
    args.img_size = None
    return args

def setup(args):
    config = CONFIGS[args.model_type]
    model = VitModel(config, args=args, num_classes=args.num_classes)
    return args, model

def train(args, model):
    # Prepare dataset
    
    
    train_loader, test_loader = args.dataset

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
                                        save_top_k=4, 
                                        monitor='valid_loss',
                                        filename=args.name + "-{epoch:02d}-{valid_loss:.2e}")
    
    epochs = args.num_steps // len(train_loader)

    print(f'Train for {epochs} epochs.')
    if len(args.device) == 1:
        trainer = Trainer(accelerator="gpu", 
                    log_every_n_steps=1,
                    devices=1,
                    max_epochs=epochs,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    precision=32,
                    logger=TensorBoardLogger("runs", name=args.name),
                    callbacks=[checkpoint_callback, lr_monitor])
    else:
        trainer = Trainer(accelerator="gpu", 
                    log_every_n_steps=1,
                    devices=args.device,
                    strategy=DDPStrategy(find_unused_parameters=False),
                    max_epochs=epochs,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    precision=32,
                    logger=TensorBoardLogger("runs", name=args.name),
                    callbacks=[checkpoint_callback, lr_monitor])
    
    # trainer.validate(model=model, dataloaders=test_loader)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    



def minimum_example():

    def generate1():
        a = np.array([np.random.randn()+3,np.random.randn()+3, np.random.randn()-3,np.random.randn()-3])
        a = (a - np.min(a))/(np.max(a)-np.min(a))

        return np.array([np.array(a).reshape(2,2)])

    def generate2():
        a = np.array([np.random.randn()-3,np.random.randn()-3, np.random.randn()+3,np.random.randn()+3])
        a = (a - np.min(a))/(np.max(a)-np.min(a))

        return np.array([np.array(a).reshape(2,2)])
    
    data_num = 5000
    inputs = np.array([ generate1() for _ in range(data_num)] + [ generate2() for _ in range(data_num)])
    outputs = np.array([ 0 for _ in range(data_num)] + [ 1 for _ in range(data_num)])


    input = torch.tensor(inputs, dtype=torch.float32)
    output = torch.tensor(outputs, dtype=torch.int64)

    dataset = torch.utils.data.TensorDataset(input, output)

    batch_size = 256
    train_test_split = 0.8
    total = len(dataset)
    train_size = floor(total*train_test_split)
    test_size = total - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,drop_last=False, num_workers=32, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,drop_last=False, num_workers=32, pin_memory=True)

    

    args = get_default_args()

    args.model_type = 'ViT-minimum'
    args.num_classes = 2
    args.dataset = (train_loader, valid_loader)
    args.name = 'Vit_minimum'
    args.num_steps = 1000000
    args.device = [0,1]
    args.gradient_accumulation_steps = 1
    args.img_size = 2
    args.learning_rate = 1e-3
    args.weight_decay = 0
    args.decay_type = 'cosine'
    args.warmup_steps = 500

    args, model = setup(args)


    train(args, model)



def minimum_example_with_rotate():

    # 4 rotated ones as first class, diagonal ones as second class

    def generate1():
        a = np.array([np.random.randn()+3,np.random.randn()+3, np.random.randn()-3,np.random.randn()-3])
        a = (a - np.min(a))/(np.max(a)-np.min(a))

        return np.array([np.array(a).reshape(2,2)])

    def generate2():
        a = np.array([np.random.randn()+3,np.random.randn()-3, np.random.randn()-3,np.random.randn()+3])
        a = (a - np.min(a))/(np.max(a)-np.min(a))

        return np.array([np.array(a).reshape(2,2)])

    data_num = 60000
    inputs = np.array([ generate1() for _ in range(data_num//4)] +
                        [np.rot90(generate1(), k=1, axes=(1,2)) for _ in range(data_num//4)] +
                        [np.rot90(generate1(), k=2, axes=(1,2)) for _ in range(data_num//4)] +
                        [np.rot90(generate1(), k=3, axes=(1,2)) for _ in range(data_num//4)] +

                        [ generate2() for _ in range(data_num//2)]+
                    [np.rot90(generate2(), axes=(1,2)) for _ in range(data_num//2)])
                        
    outputs = np.array([ 0 for _ in range(data_num)] + [ 1 for _ in range(data_num)])




    input = torch.tensor(inputs, dtype=torch.float32)
    output = torch.tensor(outputs, dtype=torch.int64)

    dataset = torch.utils.data.TensorDataset(input, output)

    batch_size = 256
    train_test_split = 0.8
    total = len(dataset)
    train_size = floor(total*train_test_split)
    test_size = total - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,drop_last=False, num_workers=32, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,drop_last=False, num_workers=32, pin_memory=True)

    

    args = get_default_args()

    args.model_type = 'ViT-minimum_rotate'
    args.num_classes = 2
    args.dataset = (train_loader, valid_loader)
    args.name = 'ViT-minimum_rotate'
    args.num_steps = 1000000
    args.device = [0,1]
    args.gradient_accumulation_steps = 1
    args.img_size = 2
    args.learning_rate = 1e-4
    args.weight_decay = 0
    args.decay_type = 'cosine'
    args.warmup_steps = 500

    args, model = setup(args)

    # trained = VitModel.load_from_checkpoint('')

    # model.model.transformer.embeddings.patch_embeddings.weight.data = 

    train(args, model)


