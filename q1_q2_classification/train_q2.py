import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
       
        self.resnet.fc = nn.Linear(512, num_classes)
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return unnormalized log-probabilities here
        return self.resnet(x)
	
        #                          END OF YOUR CODE                      #
        ##################################################################


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 50 in 50 epochs
    ##################################################################
    # args = ARGS(
    #     epochs=50,
    #     inp_size=64,
    #     use_cuda=True,
    #     val_every=70
    #     lr=# TODO,
    #     batch_size=#TODO,
    #     step_size=#TODO,
    #     gamma=#TODO
    # )
    
    args = ARGS(
        epochs=50,
        inp_size=64,
        use_cuda=True,
        val_every=70,
        lr=0.001,
        batch_size=32,
        step_size=10,
        gamma=0.85
    )
    
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # trains model using your training code and reports test mAP
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)

    torch.save({
        'model_states':model.state_dict(),
        'optimizer_states':optimizer.state_dict(),
        }, 'Q2Model_checkpoint.pth')
