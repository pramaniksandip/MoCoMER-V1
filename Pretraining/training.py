import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils import DataPreperation
from DataModule import DataModule
from Model import Model
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from utils import ScaleAugmentation, ScaleToLimitRange, GaussianBlur, Solarization

#### Image H-Params ####
K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024
#### ####

def collate_fn(batch):
    orgpic1_batch = [item[0] for item in batch]
    orgpic2_batch = [item[1] for item in batch]

    heights_x1 = [s.size(1) for s in orgpic1_batch]
    widths_x1 = [s.size(2) for s in orgpic1_batch]
    heights_x2 = [s.size(1) for s in orgpic2_batch]
    widths_x2 = [s.size(2) for s in orgpic2_batch]

    n_samples = len(orgpic1_batch)
    max_height_x1 = max(heights_x1)
    max_width_x1 = max(widths_x1)
    max_height_x2 = max(heights_x2)
    max_width_x2 = max(widths_x2)

    max_height = max(max_height_x1, max_height_x2)
    max_width = max(max_width_x1, max_width_x2)

    x1 = torch.zeros(n_samples, 1, max_height, max_width)
    x2 = torch.zeros(n_samples, 1, max_height, max_width)
    img_mask = torch.ones(n_samples, max_height, max_width, dtype=torch.bool)

    for idx, (s_x1, s_x2) in enumerate(zip(orgpic1_batch, orgpic2_batch)):
        x1[idx, :, :heights_x1[idx], :widths_x1[idx]] = s_x1
        x2[idx, :, :heights_x2[idx], :widths_x2[idx]] = s_x2
        img_mask[idx, :max(heights_x1[idx], heights_x2[idx]), :max(widths_x1[idx], widths_x2[idx])] = 0



    return x1, x2, img_mask

class NumpyToPIL:
    def __call__(self, img: np.ndarray) -> Image.Image:
        return Image.fromarray(img)


class PILToNumpy:
    def __call__(self, img: Image.Image) -> np.ndarray:
        return np.array(img)


cv2_transforms = transforms.Compose([
    ScaleAugmentation(K_MIN, K_MAX),
    ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)
])

pil_transforms = transforms.Compose([
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.2)],
        p=0.5
    ),
    #GaussianBlur(p=1.0),
    #Solarization(p=0.0),
    transforms.ToTensor(),
])


batch_size = 8
epochs = 200

root_path = "train_data/"
model_path = "MODEL/"

data = DataPreperation(root_path=root_path)

train_df = data.Train_Test_Data()

dm = DataModule(bs=16,root_path=root_path,df=train_df,cv2_transforms=cv2_transforms,pil_transforms=pil_transforms)

#train_loader = DataLoader(dm,batch_size=batch_size,drop_last = True, shuffle = True)
train_loader = DataLoader(dm,batch_size=batch_size,drop_last = True, shuffle = True, collate_fn=collate_fn)

#DenseNet Hyper-Parameters
growth_rate=24
num_layers=16
reduction=0.5


#Hyper-parameter

arch='DenseNet'
bn_splits=8
cos=True
k=200
knn_t=0.1
moco_dim=128
moco_k=4096
momentum=0.99
moco_t=0.1
results_dir='/results'
resume=''
schedule=[]
symmetric=False
wd=0.0005
m = 4096
temperature = 0.5

##Training

# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, mask in train_bar:
        x_q, x_k, mask= x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True), mask.cuda(non_blocking=True)
        _,query,_ = encoder_q(x_q,mask)

        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _,key,_ = encoder_k(x_k[idx],mask[idx])
        key = key[torch.argsort(idx)]

        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

##Training Loop

model_q = Model(growth_rate=growth_rate,num_layers=num_layers,reduction=reduction).cuda()
model_k = Model(growth_rate=growth_rate,num_layers=num_layers,reduction=reduction).cuda()

for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
    param_k.data.copy_(param_q.data)
        # not update by gradient
    param_k.requires_grad = False
    
optimizer = optim.SGD(model_q.parameters(), lr=1e-3, weight_decay=1e-6)

# c = len(memory_data)

memory_queue = F.normalize(torch.randn(m, 512).cuda(), dim=-1)

train_losses = []
save_name_pre = 'MoCo_Backbone_DensNet_SATD_AUG'
best_acc = 0.0
for epoch in range(1, epochs + 1):
    train_loss = train(model_q, model_k, train_loader, optimizer)
    train_losses.append(train_loss)



torch.save(model_q.backbone.state_dict(), model_path+'{}_model.pth'.format(save_name_pre))


with open("loss.txt", "w") as output:
    output.write(str(train_losses))