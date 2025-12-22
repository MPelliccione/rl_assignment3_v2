import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        #select device
        if device == torch.device('cpu') and torch.cuda.is_available():
            device = torch.device('cuda')
        self.device = device

        #HYPERPARAMETERS
        self.gamma = 0.99
        self.lr = 1e-3
        self.lambda_gae = 0.95
        self.delta = 0.01

        #CNN
        def cnn_block():
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
        
        #First Head: Policy
        self.cnn_actor = cnn_block().to(self.device)
        self.actor_mean = nn.Linear(1024, 3).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, 3)).to(self.device)

        #Second Head: Value
        self.cnn_critic = cnn_block().to(self.device)
        self.critic_head = nn.Linear(1024, 1).to(self.device)

        #OPTIMIZER - Adam for critic
        self.critic_optimizer = torch.optim.Adam(
            list(self.cnn_critic.parameters()) + list(self.critic_head.parameters()), 
            lr=self.lr
        )
        

    def forward(self, x):
        # TODO
        return x
    
    def act(self, state):
        # TODO
        return 

    def train(self):
        # TODO
        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
