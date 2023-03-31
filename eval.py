from hw2_319003323_train import ColoredMNIST, Generator
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

train_dataset = ColoredMNIST(env='train')
test_dataset = ColoredMNIST(env='test')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=12, shuffle=False)

# coloredMNIST Dataset domainB
train_dataset2 = ColoredMNIST(env='train', domainA=False)
test_dataset2 = ColoredMNIST(env='test', domainA=False)
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=64, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset2, batch_size=12, shuffle=False)

GA2B = Generator()
file1=torch.load('GA2B.pkl')
GA2B.load_state_dict(file1)

GB2A = Generator()
file2=torch.load('GB2A.pkl')
GB2A.load_state_dict(file2)


def inference(GA2B, GB2A, imgs):
    """show a generated sample from the test set"""
    GA2B.eval()
    GB2A.eval()
    realA = imgs[0][0].type(torch.Tensor)
    fakeB = GA2B(realA).detach()
    realB = imgs[1][0].type(torch.Tensor)
    fakeA = GB2A(realB).detach()

    # Arange images along x-axis
    realA = make_grid(realA, nrow=12, normalize=True)
    fakeB = make_grid(fakeB, nrow=12, normalize=True)
    realB = make_grid(realB, nrow=12, normalize=True)
    fakeA = make_grid(fakeA, nrow=12, normalize=True)

    # Arange images along y-axis
    image_gridA = torch.cat((realA, fakeA), 1)
    image_gridB = torch.cat((realB, fakeB), 1)
    plt.imshow(image_gridA.permute(1,2,0))
    plt.title('Real A vs Fake A')
    plt.axis('off')
    plt.show()
    plt.imshow(image_gridB.permute(1,2,0))
    plt.title('Real B vs Fake B')
    plt.axis('off')
    plt.show()

i= 0
for imgs in zip(test_loader,test_loader2):
  if i<6:
    inference(GA2B, GB2A, imgs)
    i+=1
  else:
    break

