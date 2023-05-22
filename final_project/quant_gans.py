import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from gaussianize import *
import torch.optim as optim
from tqdm import tqdm

# compute rolling windows from timeseries
def rolling_window(x, k, sparse=True):
    
    out = np.full([k, *x.shape], np.nan)
    N = len(x)
    for i in range(k):
        out[i, :N-i] = x[i:]
            
    if not sparse:
        return out

    return out[:, :-(k-1)]


class Loader32():
    def __init__(self, data, length):
        assert len(data) >= length
        self.data = data
        self.length = length
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.length]).reshape(-1, self.length).to(torch.float32)
        
    def __len__(self):
        return max(len(self.data)-self.length, 0)


"""
This is a custom module that trims the input tensor along the time dimension
(or any dimension specified) to avoid unnecessary computations. It is used in conjunction
with dilated convolutions to create a causal convolution​
"""
class Trim1d(nn.Module):
    def __init__(self, trim_size):
        super(Trim1d, self).__init__()
        self.trim_size = trim_size

    def forward(self, x):
        return x[:, :, :-self.trim_size].contiguous()


"""
This is the core module in the Temporal Convolutional Network (TCN).
It includes two dilated causal convolution layers, each followed by a Trim1d module
(to ensure the output size matches the input size), a ReLU activation, and a dropout layer.
The entire sequence is wrapped into a PyTorch Sequential model. The block also includes 
a residual layer, which is used when the number of input channels doesn't match the number 
of output channels, to ensure the residual connection can be added correctly. 
The forward method calculates both the output of the sequential model and the residual connection,
and returns their sum after applying a ReLU activation
"""
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, dilation=dilation, padding=padding))
        
        # Second convolution layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, dilation=dilation, padding=padding))
        
        self.trim = Trim1d(padding)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # self.net = nn.Sequential(self.conv1, self.relu, self.dropout, self.conv2, self.relu, self.dropout)
        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu, self.dropout, self.conv2, self.relu, self.dropout)
        else:
            self.net = nn.Sequential(self.conv1, self.trim, self.relu, self.dropout, self.conv2, self.trim, self.relu, self.dropout)

        # Residual connection
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.5)
        self.conv2.weight.data.normal_(0, 0.5)
        if self.residual is not None:
            self.residual.weight.data.normal_(0, 0.5)

    def forward(self, x):
        out = self.net(x)
        residual = x if self.residual is None else self.residual(x)
        return out, self.relu(out + residual)

"""
Generator and Discriminator classes: These are the main parts of the model.
Both of them use a sequence of TemporalBlocks followed by a final Trim1d layer.
In the forward method, they calculate the output for each TemporalBlock, storing
the 'skip' outputs in a list. Then, they add the final output to the sum of the skip outputs
and pass the result through the final Trim1d layer. The generator has a sequence 
of TemporalBlocks with varying dilations and paddings, while the discriminator has 
a fixed dilation and padding for all TemporalBlocks
"""
class Generator(nn.Module):
    def __init__(self):        
        super(Generator, self).__init__()
        self.network = nn.ModuleList([TemporalBlock(3, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.final_conv = nn.Conv1d(80, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.network:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.final_conv(x + sum(skip_layers))
        return x
    

class Discriminator(nn.Module):
    def __init__(self, seq_len, conv_dropout=0.05):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.network = nn.ModuleList([TemporalBlock(1, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.final_conv = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.network:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.final_conv(x + sum(skip_layers))
        if x.size(0) != self.seq_len:
            target = torch.zeros(self.seq_len, 1, 1)
            target[:x.size()[0], :x.size()[1], :x.size()[2]] = x
            x = target
        return self.to_prob(x.T).squeeze()


def main():
    relative_path = './final_project/'
    data_path = 'AAPL.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def dateparse(d):
        return pd.Timestamp(d)

    data = pd.read_csv(relative_path + data_path, parse_dates={'datetime': ['Date']}, date_parser=dateparse)
    df = data['Close']

    returns = df.shift(1)/df - 1
    log_returns = np.log(df/df.shift(1))[1:].to_numpy().reshape(-1, 1)
    standardScaler1 = StandardScaler()
    standardScaler2 = StandardScaler()
    gaussianize = Gaussianize()
    log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))

    num_epochs = 100
    nz = 3
    batch_size = 80
    seq_len = 80
    clip= 0.01
    lr = 0.0002
    generator_path = f'./trained/'
    file_name = 'AAPL_daily'

    generator = Generator().to(device)

    train = False

    if train:
        discriminator = Discriminator(seq_len).to(device)
        disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)
        gen_optimizer = optim.RMSprop(generator.parameters(), lr=lr)

        dataset = Loader32(log_returns_preprocessed, 127)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        dataset = Loader32(log_returns_preprocessed, 1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        t = tqdm(range(num_epochs))
        for epoch in t:
            for idx, data in enumerate(dataloader, 0):

                discriminator.zero_grad()
                real = data.to(device)
                batch_size, seq_len = real.size(0), real.size(1)
                noise = torch.randn(batch_size, nz, seq_len, device=device)
                fake = generator(noise).detach()
                disc_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
                disc_loss.backward()
                disc_optimizer.step()

                for dp in discriminator.parameters():
                    dp.data.clamp_(-clip, clip)
        
                if idx % 5 == 0:
                    generator.zero_grad()
                    gen_loss = -torch.mean(discriminator(generator(noise)))
                    gen_loss.backward()
                    gen_optimizer.step()            
            t.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))
                
        # Save
        torch.save(generator, f'{relative_path + generator_path}trained_generator_{file_name}_epoch_{epoch}.pth')

    else:
        # Load
        generator = torch.load(f'{relative_path + generator_path}trained_generator_{file_name}_epoch_{num_epochs-1}.pth')
        generator.eval() 


if __name__ == "__main__":
    main()
	