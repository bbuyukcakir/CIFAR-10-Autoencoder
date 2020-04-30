from torch import nn
import torch
import matplotlib.pyplot as plt
import os
import random
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as trans
from PIL import Image
import matplotlib.cm as cm

random.seed(42)


class Generator:

    def __init__(self, path):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.path = os.path.join(dir_path, path)
        self.files = os.listdir(self.path)
        self.filecount = 0
        self.batch_count = 0

    def load(self, img_num, set_count=False):
        # image = Image.open()
        image = plt.imread(os.path.join(self.path, self.files[img_num - 1]))
        if set_count:
            self.filecount = img_num
        # image = (image[:,:,0] + image[:,:,0] +image[:,:,0]) / 3
        return torch.tensor(image).view(3, 32, 32)

    def load_batch(self, batch_size):
        if self.batch_count == 0:
            self._generate_batches(batch_size)
        batch = torch.zeros(batch_size, 3, 32, 32)

        for n, i in enumerate(self.batch_indices[self.batch_count]):
            batch[n, :, :, :] = (self.load(int(i[:-4])))
        self.batch_count += 1
        return batch

    def _generate_batches(self, batch_size):
        files = self.files
        random.shuffle(files)
        n = []
        self.batch_indices = []
        for i in range(len(files) // batch_size):
            n = files[i:i + batch_size]
            self.batch_indices.append(n)
        # self.batch_indices = [files[i:i+batch_size] for i in range(batch_size + 1)]


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.device = torch.device("cuda")

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 6, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 5, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.to(self.device)

        self.enc_out = self.encoder(x)
        self.out = self.decoder(self.enc_out)
        return self.out

    def get_loss(self, true, pred):
        t = true.view(-1, 1)
        p = pred.view(-1, 1)
        l = nn.MSELoss()
        loss = l(p.to(torch.device('cuda')), t.to(torch.device('cuda')))

        # loss = torch.abs(F.kl_div(p,t))
        # l = nn.CosineSimilarity(dim = 0)
        # loss = 1 - l(pred, true).mean()
        # loss = (0.5 * (t - p) ** 2).sum()
        # l = nn.CosineSimilarity(dim=0)
        # loss = l(t, p).mean()
        return loss.to(torch.device("cuda"))


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    a = Generator("train")
    z = Autoencoder()
    # a.batch_indices
    z.cuda(device=torch.device("cuda"))
    # opt = torch.optim.Adam(z.parameters(), lr=0.0001)
    batch_size = 64
    loss_prev = 0

    # for i in range(20):
    #     for j in range(len(a.files) // batch_size):
    #         batch = a.load_batch(batch_size)
    #         x = z.forward((batch).to(torch.device("cuda")))
    #         loss = z.get_loss(batch, x)
    #         loss.backward()
    #         # with torch.no_grad():
    #         #     axs[0].imshow(batch.cpu()[0, :, :, :].view(32, 32, 3))
    #         #     axs[1].imshow(z.unconv2out.cpu()[0, :, :, :].view(32, 32, 3))
    #         #     fig.show()
    #         opt.step()
    #         opt.zero_grad()
    #         z.zero_grad()
    #
    #         if j % 10 == 9:
    #             print(f"Epoch:{i + 1} Batch: {j + 1} complete. Loss: {loss.data} (delta: {loss.data - loss_prev})")
    #             loss_prev = loss.data
    #     a.batch_count = 0
    #
    #     with torch.no_grad():
    #         fig, axs = plt.subplots(2, 3)
    #         axs[0, 0].imshow(batch.cpu()[0, :, :, :].view(32, 32, 3))
    #         axs[1, 0].imshow(z.out.cpu()[0, :, :, :].view(32, 32, 3))
    #         axs[0, 1].imshow(batch.cpu()[10, :, :, :].view(32, 32, 3))
    #         axs[1, 1].imshow(z.out.cpu()[10, :, :, :].view(32, 32, 3))
    #         axs[0, 2].imshow(batch.cpu()[30, :, :, :].view(32, 32, 3))
    #         axs[1, 2].imshow(z.out.cpu()[30, :, :, :].view(32, 32, 3))
    #         fig.show()
