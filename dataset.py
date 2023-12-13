import torch
import torch.utils.data as dutils
from torchvision.io.image import read_image
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt


class Dataset(dutils.Dataset):
	def __init__(self, subset):
		super().__init__()
		low = os.listdir('low')
		high = os.listdir('high')
		num_training = int(len(low) * 0.8)
		num_validation = int(len(low) * 0.1)

		if subset == 'training':
			self.low = low[:num_training]
			self.high = high[:num_training]
		elif subset == 'validation':
			self.low = low[num_training:num_training+num_validation]
			self.high = high[num_training:num_training+num_validation]
		elif subset == 'testing':
			self.low = low[num_training+num_validation:]
			self.high = high[num_training+num_validation:]
		else:
			print('Warning: Unknown subset, using entire dataset..')
			self.low = low
			self.high = high
			
	
	def __len__(self):
		return len(self.low)
	
	def __getitem__(self, index):
		low_img = read_image('low/' + self.low[index])[:3, :, :]
		high_img = read_image('high/' + self.high[index])[:3, :, :]
		return low_img / 255.0, high_img / 255.0
	

train_ds = Dataset('training')
test_ds = Dataset('testing')
val_ds = Dataset('validation')
train_dl = dutils.DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = dutils.DataLoader(test_ds, batch_size=32, shuffle=True)
val_dl = dutils.DataLoader(val_ds, batch_size=32, shuffle=True)

if __name__ == '__main__':
	low_imgs, high_imgs = next(iter(test_dl))

	plt.figure(figsize=(4, 8))
	for i in range(0, 8, 2):
		plt.subplot(4, 2, i+1)
		plt.imshow(T.ToPILImage()(low_imgs[i]))

		plt.subplot(4, 2, i+2)
		plt.imshow(T.ToPILImage()(high_imgs[i]))
	plt.show()