# QuantGANs for Financial Time Series Generation
This repository contains an implementation of Quantitative Generative Adversarial Networks (QuantGANs) for generating synthetic financial time series data. The model is built using the PyTorch framework and consists of a generator and a discriminator. Both the generator and the discriminator use Temporal Block layers for temporal data processing.

## Temporal Block
A Temporal Block is a custom layer designed to handle time series data effectively. It consists of two 1D convolutional layers, each followed by a PReLU activation and dropout. The output from the second layer is added to the original input to form a residual connection. The final output is then passed through a PReLU activation.

## Training the Model
To train the QuantGANs model, follow these steps:

1. Prepare the financial data by downloading a historical stock prices CSV file from Yahoo Finance. In this example, we use the adjusted closing prices of the Apple Inc. (AAPL) stock.
2. Load the financial data using the load_financial_data function.
3. Initialize the generator and the discriminator.
4. Train the QuantGANs model using the example from the example.ipynb.

## Using the Model in a Jupyter Notebook
You can use the QuantGANs model in a Jupyter Notebook by importing the necessary classes and functions from the provided code. Follow these steps:

1. Import the required classes and functions from the provided code.
2. Prepare the financial data as described in the "Training the Model" section above.
3. Initialize the generator and the discriminator.
4. Train the QuantGANs model using the the example from the example.ipynb function.
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
data_path = 'AAPL.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dateparse(d):
    return pd.Timestamp(d)

# Preprossesing
data = pd.read_csv(data_path, parse_dates={'datetime': ['Date']}, date_parser=dateparse)
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

train = True

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
    torch.save(generator, f'{generator_path}trained_generator_{file_name}_epoch_{epoch}.pth')

else:
    # Load
    generator = torch.load(f'{generator_path}trained_generator_{file_name}_epoch_{num_epochs-1}.pth')
    generator.eval() 

noise = torch.randn(80,3,127).to(device)
y = generator(noise).cpu().detach().squeeze();

y = (y - y.mean(axis=0))/y.std(axis=0)
y = standardScaler2.inverse_transform(y)
y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
y = standardScaler1.inverse_transform(y)

#filtering
y = y[(y.max(axis=1) <= 2 * log_returns.max()) & (y.min(axis=1) >= 2 * log_returns.min())]
y -= y.mean()
len(y)    
```

## Visualizing Generated Time Series Data
After generating synthetic financial time series data using the trained QuantGANs model, you can visualize the data using Python libraries like Matplotlib or Plotly. Here's an example using Matplotlib:

```python
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(np.cumsum(y[0:30], axis=1).T)
ax.set_title('30 generated log graphs'.format(len(y)))
ax.set_xlabel('days')
ax.set_ylabel('Cumalative log');

windows = [1, 5, 20, 100]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))


for i in range(len(windows)):
    row = min(max(0, i-1), 1)
    col = i % 2
    real_dist = rolling_window(log_returns, windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
    fake_dist = rolling_window(y.T, windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
    axs[row, col].hist(np.array([real_dist, fake_dist], dtype='object'), bins=50, density=True)
    axs[row,col].set_xlim(*np.quantile(fake_dist, [0.001, .999]))
    
    axs[row,col].set_title('{} day distribution'.format(windows[i]), size=16)
    axs[row,col].yaxis.grid(True, alpha=0.5)
    axs[row,col].set_xlabel('Cumalative log')
    axs[row,col].set_ylabel('Frequency')

axs[0,0].legend(['Historical', 'Synthetic'])
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