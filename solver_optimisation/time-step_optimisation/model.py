import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class CNQNet(nn.Module):
    def __init__(self, n_output):
        super().__init__()
        self.conv1 = nn.Conv1d(3,5,kernel_size=3,stride=1, padding='same')
        self.conv2 = nn.Conv1d(5,5,kernel_size=3,stride=1, padding='same')
        self.pool = nn.MaxPool1d(3, stride=2, padding=1) #1x2 maxpool
        self.fc1 = nn.Linear(16*5,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,n_output)
    
    def forward(self,x):
        #print('input', x.shape)
        x = F.elu(self.conv1(x)) #256x10
        #print('conv1', x.shape)
        x = self.pool(x) #127x10
        #print('pool',x.shape)
        x = F.elu(self.conv2(x)) #128x10
        #print(x.shape)
        x = self.pool(x) #64x10
        #print(x.shape)
        x = F.elu(self.conv2(x)) #64x10
        #print(x.shape)    
        x = self.pool(x) #32x10
        #print(x.shape)
        x = x.view(-1, 16*5) #flattening
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        #x = self.fc3(x)
        #print('action shape:', x.shape)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.MSELoss()
        self.device = device
        self.ls = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        #print('state shape:', state.shape)
        #print('action shape:', action.shape)

        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        #print('state shape:', state.shape)
        #print('action shape:', action.shape)
        #print('reward shape:', reward.shape)
        #print('done len =', len(done))

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        #print('target shape =', pred.shape)
        len_ep = state.shape[0]
        for idx in range(len_ep):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = (reward[idx] + self.gamma * self.model(next_state[idx]).max(1).values)
            #print('argmax action =', torch.argmax(action).item())
            target[idx,torch.argmax(action[idx]).item()] = Q_new
            #print(target)
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        #print('LOSS =', loss)
        self.ls = loss.cpu().detach().numpy()
        loss.backward()

        self.optimizer.step()