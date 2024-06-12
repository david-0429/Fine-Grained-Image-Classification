import argparse
import os
import random
import shutil
from os.path import join
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm
import wandb
import pdb
import matplotlib.pyplot as plt
from PIL import Image

from LabelSmoothing import LabelSmoothingLoss


#######################
##### 1 - Setting #####
#######################

##### args setting
parser = argparse.ArgumentParser()
#parser.add_argument('-d', '--dir', default='fgvc', help='dataset dir')
parser.add_argument('-b', '--batch_size', default=64, help='batch_size')
parser.add_argument(
    '-g', '--gpu', default='0', help='example: 0 or 1, to use different gpu'
)
parser.add_argument('-w', '--num_workers', default=12, help='num_workers of dataloader')
parser.add_argument('-s', '--seed', default=2020, help='random seed')
parser.add_argument(
    '-n',
    '--note',
    default='',
    help='exp note, append after exp folder, fgvc(_r50) for example',
)
parser.add_argument(
    '-a',
    '--amp',
    default=0,
    help='0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp',
)
args = parser.parse_args()

##### exp setting
seed = int(args.seed)
# datasets_dir = args.dir
nb_epoch = int(100)  # 128 as default to suit scheduler
val_interval = 5
batch_size = int(args.batch_size)
num_workers = int(args.num_workers)
lr_begin = (batch_size / 256) * 0.1  # learning rate at begining
use_amp = int(args.amp)  # use amp to accelerate training

exp_dir = 'result/'
data_sets = ['train', 'eval', 'test']
'''
##### data settings
data_dir = join('data', datasets_dir)
data_sets = ['train', 'test']
nb_class = len(
    os.listdir(join(data_dir, data_sets[0]))
)  # get number of class via img folders automatically
exp_dir = 'result/{}{}'.format(datasets_dir, args.note)  # the folder to save model
'''

##### CUDA device setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


##### Random seed setting
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


##### Dataloader setting
re_size = 300
crop_size = 224

train_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.RandomCrop(crop_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

trainset = torchvision.datasets.FGVCAircraft(root='./data', split='train',
                                        download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

evalset = torchvision.datasets.FGVCAircraft(root='./data', split='val',
                                       download=True, transform=test_transform)
eval_loader = torch.utils.data.DataLoader(evalset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

testset = torchvision.datasets.FGVCAircraft(root='./data', split='test',
                                       download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
nb_class = len(trainset.classes)
print("Number of classes in FGVCAircraft dataset:", nb_class)
'''
train_set = ImageFolder(root=join(data_dir, data_sets[0]), transform=train_transform)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
'''

#--------------------------------------------------------------#

fig_label_num = 4
fig_img_num = 5

# 모든 레이블 추출
all_labels = list(trainset.classes)

# 무작위로 10개의 고유한 레이블 선택
selected_labels = random.sample(all_labels, fig_label_num)

# 레이블별로 이미지를 하나씩 선택
label_to_images = {label: [] for label in selected_labels}
for img, label in trainset:
    label_name = trainset.classes[label]
    if label_name in selected_labels and len(label_to_images[label_name]) < fig_img_num:
        label_to_images[label_name].append(img)
    if all(len(images) == fig_img_num for images in label_to_images.values()):
        break

# 선택된 이미지와 레이블을 리스트로 변환
selected_images = [img for images in label_to_images.values() for img in images]
selected_labels = [label for label, images in label_to_images.items() for _ in images]

# 이미지 시각화 함수
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')

# 이미지와 레이블 시각화 및 파일로 저장
fig, axes = plt.subplots(fig_label_num, fig_img_num, figsize=(20, 16))
axes = axes.flatten()

for img, label, ax in zip(selected_images, selected_labels, axes):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(label)
    ax.axis('off')

# 파일로 저장
plt.savefig('fgvc_aircraft_samples.png', bbox_inches='tight')
plt.close(fig)

print("The image grid with labels has been saved as 'fgvc_aircraft_samples.png'.")

#--------------------------------------------------------------#


### wandb init
print("wandb init")
def get_timestamp():
    return datetime.now().strftime("%b%d_%H-%M-%S")
wandb.init(
    # Set the project where this run will be logged
    project=f"Team-Project", 
    name=f"{'resnet50'}_{args.batch_size}_{lr_begin}_{crop_size}-{get_timestamp()}"
)


##### Model settings
net = torchvision.models.resnet50(
    pretrained=True
)  # to use more models, see https://pytorch.org/vision/stable/models.html
net.fc = nn.Linear(
    net.fc.in_features, nb_class
)  # set fc layer of model with exact class number of current dataset

for param in net.parameters():
    param.requires_grad = True  # make parameters in model learnable


##### optimizer setting
LSLoss = LabelSmoothingLoss(
    classes=nb_class, smoothing=0.1
)  # label smoothing to improve performance
optimizer = torch.optim.SGD(
    net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epoch)

'''
##### file/folder prepare
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

shutil.copyfile('train.py', exp_dir + '/train.py')
shutil.copyfile('LabelSmoothing.py', exp_dir + '/LabelSmoothing.py')

with open(os.path.join(exp_dir, 'train_log.csv'), 'w+') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc, Test_Acc\n')
'''

##### Apex
if use_amp == 1:  # use nvidia apex.amp
    print('\n===== Using NVIDIA AMP =====')
    from apex import amp

    net.cuda()
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using NVIDIA AMP =====\n')
elif use_amp == 2:  # use torch.cuda.amp
    print('\n===== Using Torch AMP =====')
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using Torch AMP =====\n')


########################
##### 2 - Training #####
########################
net.cuda()
min_train_loss = float('inf')
max_eval_acc = 0

for epoch in range(nb_epoch):
    print('\n===== Epoch: {} ====='.format(epoch))
    net.train()  # set model to train mode, enable Batch Normalization and Dropout
    lr_now = optimizer.param_groups[0]['lr']
    train_loss = train_correct = train_total = idx = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, ncols=80)):
        idx = batch_idx

        if inputs.shape[0] < batch_size:
            continue

        optimizer.zero_grad()  # Sets the gradients to zero
        inputs, targets = inputs.cuda(), targets.cuda()

        ##### amp setting
        if use_amp == 1:  # use nvidia apex.amp
            x = net(inputs)
            loss = LSLoss(x, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        elif use_amp == 2:  # use torch.cuda.amp
            with autocast():
                x = net(inputs)
                loss = LSLoss(x, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x = net(inputs)
            loss = LSLoss(x, targets)
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(x.data, 1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets.data).cpu().sum()
        train_loss += loss.item()

    scheduler.step()

    train_acc = 100.0 * float(train_correct) / train_total
    train_loss = train_loss / (idx + 1)
    print(
        'Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f}% ({}/{})'.format(
            lr_now, train_loss, train_acc, train_correct, train_total
        )
    )
    wandb.log({"epoch/train_acc": train_acc, "epoch/trn_loss": train_loss, "epoch": epoch})
    
    if epoch % val_interval == 0:
        ##### Evaluating model with test data every epoch
        with torch.no_grad():
            net.eval()  # set model to eval mode, disable Batch Normalization and Dropout
            '''
            eval_set = ImageFolder(
                root=join(data_dir, data_sets[-1]), transform=test_transform
            )
            eval_loader = DataLoader(
                eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            '''
            eval_correct = eval_total = 0
            for _, (inputs, targets) in enumerate(tqdm(eval_loader, ncols=80)):
                inputs, targets = inputs.cuda(), targets.cuda()
                x = net(inputs)
                _, predicted = torch.max(x.data, 1)
                eval_total += targets.size(0)
                eval_correct += predicted.eq(targets.data).cpu().sum()
            eval_acc = 100.0 * float(eval_correct) / eval_total
            print(
                '{} | Acc: {:.3f}% ({}/{})'.format(
                    data_sets[1], eval_acc, eval_correct, eval_total
                )
            )
            wandb.log({"epoch/val_acc": eval_acc, "epoch": epoch})

            if epoch + 1 % 20 == 0:
              softmax = torch.nn.Softmax(dim=1)
              prob = softmax(x)
              std = torch.std(prob, dim=1)
              std_mean = torch.mean(std)
              print(x[0])
              print(
                '{} | std_mean: {:.3f}'.format(
                    data_sets[1], std_mean
                )
              )
              wandb.log({"prob": x[0], "epoch": epoch})
              wandb.log({"epoch/std_mean": std_mean, "epoch": epoch})

            '''
            ##### Logging
            with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
                file.write(
                    '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%\n'.format(
                        epoch, lr_now, train_loss, train_acc, eval_acc
                    )
                )
            
            ##### save model with highest acc
            if eval_acc > max_eval_acc:
                max_eval_acc = eval_acc
                torch.save(
                    net.state_dict(),
                    os.path.join(exp_dir, 'max_acc.pth'),
                    _use_new_zipfile_serialization=False,
                )
            '''

#--------------------------------------------------------------#

# Forward hook을 사용하여 특징 맵 추출
feature_maps = {}

def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output
    return hook

# 첫 번째, 중간, 마지막 레이어에 hook 등록
hooks = []
hooks.append(net.conv1.register_forward_hook(hook_fn('conv1')))
hooks.append(net.layer2[-1].conv3.register_forward_hook(hook_fn('layer2')))
hooks.append(net.layer4[-1].conv3.register_forward_hook(hook_fn('layer4')))

# 검증 배치의 0번째 이미지 가져오기
val_batch = next(iter(eval_loader))
val_image = val_batch[0][0].unsqueeze(0).cuda()

# 모델을 통해 이미지 전달 (순전파)
net.eval()
with torch.no_grad():
    net(val_image)

# hook 제거
for hook in hooks:
    hook.remove()

# 폴더 생성
output_folder = 'feature_maps'
os.makedirs(output_folder, exist_ok=True)

# 특징 맵 시각화 및 파일 저장 함수
def save_feature_map(feature_map, file_name):
    feature_map = feature_map.cpu().squeeze()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(feature_map.mean(dim=0).detach().numpy(), cmap='viridis')
    ax.axis("off")
    plt.savefig(os.path.join(output_folder, file_name), bbox_inches='tight')
    plt.close(fig)

# 특징 맵 저장
save_feature_map(feature_maps['conv1'], 'conv1_feature_map.png')
save_feature_map(feature_maps['layer2'], 'layer2_feature_map.png')
save_feature_map(feature_maps['layer4'], 'layer4_feature_map.png')

# 원본 이미지 저장
original_image = val_image.cpu().squeeze().permute(1, 2, 0).numpy()
original_image = original_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
original_image = np.clip(original_image, 0, 1)
plt.imshow(original_image)
plt.axis('off')
plt.savefig(os.path.join(output_folder, 'original_image.png'), bbox_inches='tight')
plt.close()

print("Feature maps and original image saved successfully.")

#--------------------------------------------------------------#



'''
########################
##### 3 - Testing  #####
########################
print('\n\n===== TESTING =====')

with open(os.path.join(exp_dir, 'train_log.csv'), 'a') as file:
    file.write('===== TESTING =====\n')

##### load best model
net.load_state_dict(torch.load(join(exp_dir, 'max_acc.pth')))
net.eval()  # set model to eval mode, disable Batch Normalization and Dropout

for data_set in data_sets:
    
    testset = ImageFolder(
        root=os.path.join(data_dir, data_set), transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loss = correct = total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(test_loader, ncols=80)):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = net(inputs)
            _, predicted = torch.max(x.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    test_acc = 100.0 * float(correct) / total
    print('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))
    wandb.log({"epoch/test_acc": test_acc, "epoch": epoch})
    
    
    ##### logging
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

    with open(
        os.path.join(exp_dir, 'acc_{}_{:.2f}'.format(data_set, test_acc)), 'a+'
    ) as file:
        # save accuracy as file name
        pass
    
'''
wandb.finish()
