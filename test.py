import torch
from dataset import test_dl
import torchvision.transforms as T
import matplotlib.pyplot as plt

model = torch.load('models/model.pkl')

low, high = next(iter(test_dl))

with torch.no_grad():
	predictions = model(low)

plt.figure(figsize=(4, 10))
for i in range(0, 12, 3):
	plt.subplot(4, 3, i+1)
	plt.imshow(T.ToPILImage()(low[i]))

	plt.subplot(4, 3, i+2)
	plt.imshow(T.ToPILImage()(high[i]))

	plt.subplot(4, 3, i+3)
	plt.imshow(T.ToPILImage()(predictions[i]))
plt.show()
