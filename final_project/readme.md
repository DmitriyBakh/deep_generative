# QuantGANs for Financial Time Series Generation
This repository contains an implementation of Quantitative Generative Adversarial Networks (QuantGANs) for generating synthetic financial time series data. The model is built using the PyTorch framework and consists of a generator and a discriminator. Both the generator and the discriminator use Temporal Block layers for temporal data processing.

## Temporal Block
A Temporal Block is a custom layer designed to handle time series data effectively. It consists of two 1D convolutional layers, each followed by a PReLU activation and dropout. The output from the second layer is added to the original input to form a residual connection. The final output is then passed through a PReLU activation.

## Training the Model
To train the QuantGANs model, follow these steps:

1. Prepare the financial data by downloading a historical stock prices CSV file from Yahoo Finance. In this example, we use the adjusted closing prices of the Apple Inc. (AAPL) stock.
2. Load the financial data using the load_financial_data function.
3. Initialize the generator and the discriminator.
4. Train the QuantGANs model using the train_quant_gans function with the loaded data, generator, and discriminator.

## Using the Model in a Jupyter Notebook
You can use the QuantGANs model in a Jupyter Notebook by importing the necessary classes and functions from the provided code. Follow these steps:

1. Import the required classes and functions from the provided code.
2. Prepare the financial data as described in the "Training the Model" section above.
3. Initialize the generator and the discriminator.
4. Train the QuantGANs model using the train_quant_gans function.
5. After training, you can use the generator to create synthetic financial time series data by providing random noise as input.

## Getting Stock Data from Yahoo Finance
To download historical stock prices from Yahoo Finance:

1. Go to Yahoo Finance.
2. Search for the stock you are interested in (e.g., "AAPL" for Apple Inc.).
3. On the stock's summary page, click on the "Historical Data" tab.
4. Adjust the time period and frequency as desired, then click on "Apply".
5. Click on the "Download" button to download the stock data as a CSV file.

## Example
Here's a quick example of how to use the QuantGANs model in a Jupyter Notebook:

```python
import torch
from quant_gans import Generator, Discriminator, load_financial_data, train_quant_gans

# Load financial data
data = load_financial_data('AAPL.csv')

# Initialize the generator and the discriminator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(n_layers=8, n_channels=64, n_input=1, n_output=1, kernel_size=2, stride=1, dilation=2, padding=1, dropout=0.2).to(device)
discriminator = Discriminator(n_layers=8, n_channels=64, n_input=1, n_output=1, kernel_size=2, stride=1, dilation=2, padding=1, dropout=0.2).to(device)

# Train the QuantGANs model
train_quant_gans(data, generator, discriminator, num_epochs=200, batch_size=64, device=device)

#After training, you can use the generator to create synthetic financial time series data:
noise = torch.randn(batch_size, 1, 100).to(device)
generated_data = generator(noise)
```

## Visualizing Generated Time Series Data
After generating synthetic financial time series data using the trained QuantGANs model, you can visualize the data using Python libraries like Matplotlib or Plotly. Here's an example using Matplotlib:

```python
import matplotlib.pyplot as plt

# Generate synthetic financial time series data
noise = torch.randn(1, 1, 100).to(device)
generated_data = generator(noise).detach().cpu().numpy().flatten()

# Plot the generated data
plt.plot(generated_data)
plt.title("Generated Financial Time Series Data")
plt.xlabel("Time")
plt.ylabel("Normalized Adjusted Close Price")
plt.show()
```

```python
##Saving and Loading Trained Models
You can save the trained generator and discriminator models to disk for later use. To do this, you can use the torch.save function. Here's an example:

# Save the trained generator and discriminator models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

#To load the saved models back into memory, use the torch.load function:
# Load the saved generator and discriminator models
generator.load_state_dict(torch.load('generator.pth'))
discriminator.load_state_dict(torch.load('discriminator.pth'))
```