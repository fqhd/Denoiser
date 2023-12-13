from network import *
import torch
from torch import nn
from tqdm import tqdm
from dataset import train_dl, val_dl, test_dl
import matplotlib.pyplot as plt

device = (
	'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f'Using {device}')

model = U_Net(img_ch=3, output_ch=3)
model = model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train(dl):
	avg_loss = 0
	for low, high in tqdm(dl):
		low = low.to(device)
		high = high.to(device)
		prediction = model(low)
		loss = loss_fn(prediction, high)
		model.zero_grad()
		loss.backward()
		optimizer.step()
		avg_loss += loss.item()
	avg_loss /= len(dl)
	return avg_loss

def test(dl):
	avg_loss = 0
	for low, high in tqdm(dl):
		low = low.to(device)
		high = high.to(device)
		with torch.no_grad():
			prediction = model(low)
		loss = loss_fn(prediction, high)
		avg_loss += loss.item()
	avg_loss /= len(dl)
	return avg_loss

train_losses = []
val_losses = []

for i in range(20):
	train_loss = train(train_dl)
	val_loss = test(val_dl)
	print('---------------------------------------------------')
	print(f'Epoch {i+1} results:')
	print(f'Training Loss: {train_loss}')
	print(f'Validation Loss: {val_loss}')
	train_losses.append(train_loss)
	val_losses.append(val_losses)

print('Training Finished')
print('Testing model...')
test_loss = test(test_dl)
print(f'Test Loss: {test_loss}')

model = model.to('cpu')
torch.save(model, 'models/model.pkl')
print('Successfully saved model')
