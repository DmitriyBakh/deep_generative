import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import pandas as pd
import numpy as np
# import os


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        # Store the input as residual
        residual = x
        
        # Apply first convolutional layer, ReLU activation, and dropout
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        
        # Apply second convolutional layer, ReLU activation, and dropout
        if out.shape[2] == 1:
            # self.conv2.kernel_size = (1,)
            test = 1
        
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        
        # If the residual connection is needed, apply it
        if self.residual is not None:
            residual = self.residual(x)
        
        # Add the residual back to the output
        return out + residual


class Generator(nn.Module):
    def __init__(self, n_layers, n_channels, n_input, n_output, kernel_size, stride, dilation, padding, dropout):
        super(Generator, self).__init__()
        layers = []
        for i in range(n_layers):
            dilation_size = dilation ** i
            in_channels = n_input if i == 0 else n_channels
            out_channels = n_channels
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride, dilation_size, padding, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(n_channels, n_output, kernel_size=1)

    def forward(self, x):
        out = self.network(x)
        out = self.final_conv(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_layers, n_channels, n_input, n_output, kernel_size, stride, dilation, padding, dropout):
        super(Discriminator, self).__init__()
        layers = []
        for i in range(n_layers):
            dilation_size = dilation ** i
            in_channels = n_input if i == 0 else n_channels
            out_channels = n_channels
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride, dilation_size, padding, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(n_channels, n_output, 1)

    def forward(self, x):
        out = self.network(x)
        out = self.final_conv(out)
        return out	

def load_financial_data(filename):
	data = pd.read_csv(filename)
	adj_close_prices = data['Adj Close'].dropna()
	normalized_prices = adj_close_prices / adj_close_prices.iloc[0] - 1
	data_array = normalized_prices.values.reshape(-1, 1)
	return data_array

def train_quant_gans(data, generator, discriminator, num_epochs, batch_size, device):
	criterion = nn.BCEWithLogitsLoss()
	generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
	discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))	

	real_labels = torch.ones(batch_size, 1).to(device)
	fake_labels = torch.zeros(batch_size, 1).to(device)
	fixed_noise = torch.randn(batch_size, 1, 100).to(device)

	for epoch in range(num_epochs):
		for i in range(0, len(data), batch_size):
			# Train the discriminator on real data
			real_data = torch.FloatTensor(data[i:i + batch_size]).unsqueeze(1).to(device)
			real_output = discriminator(real_data)
			real_loss = criterion(real_output, real_labels)
			real_score = torch.sigmoid(real_output)

			# Train the discriminator on fake data
			noise = torch.randn(batch_size, 1, 100).to(device)
			fake_data = generator(noise)
			fake_output = discriminator(fake_data.detach())
			fake_loss = criterion(fake_output, fake_labels)
			fake_score = torch.sigmoid(fake_output)

			# Optimize the discriminator
			discriminator_loss = real_loss + fake_loss
			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()

			# Train the generator
			fake_output = discriminator(fake_data)
			generator_loss = criterion(fake_output, real_labels)

			# Optimize the generator
			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

		print(f"Epoch [{epoch + 1}/{num_epochs}] - D Loss: {discriminator_loss.item()} - G Loss: {generator_loss.item()}")


def main():
	data = load_financial_data('AAPL.csv')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	n_layers = 3
	n_channels = 11
	n_input = 1
	n_output = 1
	kernel_size = 2
	stride = 1
	dilation = 2
	padding = 1
	dropout = 0.2
	num_epochs = 200
	batch_size = 64

	generator = Generator(n_layers, n_channels, n_input, n_output, kernel_size, stride, dilation, padding, dropout).to(device)
	discriminator = Discriminator(n_layers, n_channels, n_input, n_output, kernel_size, stride, dilation, padding, dropout).to(device)

	train_quant_gans(data, generator, discriminator, num_epochs, batch_size, device)


if __name__ == "__main__":
	main()
	