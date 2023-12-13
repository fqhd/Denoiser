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
	if i == 0:
		plt.title('Input Image')
	plt.axis('off')
	plt.imshow(T.ToPILImage()(low[i]))

	plt.subplot(4, 3, i+2)
	if i == 0:
		plt.title('Ground Truth')
	plt.axis('off')
	plt.imshow(T.ToPILImage()(high[i]))

	plt.subplot(4, 3, i+3)
	if i == 0:
		plt.title('AI Denoised')
	plt.axis('off')
	plt.imshow(T.ToPILImage()(predictions[i]))
plt.show()
