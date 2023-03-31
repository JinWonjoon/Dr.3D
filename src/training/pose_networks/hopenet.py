import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision
from src.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from src.camera_utils import sample, fix_intrinsics
from scipy.spatial.transform import Rotation as R
import numpy as np
import pdb

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

class Posenet(nn.Module):
    def __init__(self, pretrained_path=False, alpha=1.0):
        super(Posenet, self).__init__()
        
        self.hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        if pretrained_path is not False:
            saved_state_dict = torch.load(pretrained_path)
            self.hopenet.load_state_dict(saved_state_dict)
        
        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor)
        
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.softmax = nn.Softmax()
        self.alpha = alpha
        self.intrinsics = fix_intrinsics()
    
    def forward(self, x):
        device = x.device
        idx_tensor = self.idx_tensor.to(device)
        
        yaw_predicted, pitch_predicted, roll_predicted = self.hopenet(x) # [yaw, pitch, roll]. degree
        yaw_predicted = self.softmax(yaw_predicted)
        pitch_predicted = self.softmax(pitch_predicted)
        roll_predicted = self.softmax(roll_predicted)
        
        yaw_predicted = (torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99)
        pitch_predicted = (torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99)
        roll_predicted = (torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99)

        return torch.stack([yaw_predicted, pitch_predicted, roll_predicted]).T * math.pi/180 # batch x 3. rad

    def get_label_from_pose(self, pose):
        """
            Args.
                pose: batch x 3. degree. tensor. (yaw, pitch, roll)
            Return.
                label: batch x 25. tensor. stacking extrinsics and normalized intrinsics.
            # camera position in cannonical view : yaw = math.pi/2, pitch : math.pi/2
            # since pose is a head pose, camera should move opposite position
        """
        
        poses = pose.cpu().detach().numpy()
                
        label = []
        for pose in poses:
            extrinsic = sample(yaw = math.pi / 2 - pose[0], pitch = math.pi / 2 - pose[1])
            r = R.from_matrix(extrinsic[:3, :3])
            rot_rad = r.as_euler('zyx') #roll, pitch, yaw
            new_rot_rad = R.from_euler('zyx', np.array([rot_rad[0] - pose[2], rot_rad[1], rot_rad[2]]))
            new_rot_mat = new_rot_rad.as_matrix()
            extrinsic[:3, :3] = new_rot_mat
            label.append(np.concatenate([extrinsic.reshape(-1), self.intrinsics.reshape(-1)]))
        label = torch.tensor(np.stack(label))
        return label
        

    def get_pose_label(self, x):
        """
            Args:
            1) x: real images.
            
            Return:
            1) pose: (yaw, pitch, roll). batch x 3. radian.
            2) label: pose condition label. batch x 25.
        """
        pose = self.forward(x) # batch x 3. rad
        label = self.get_label_from_pose(pose) # batch x 25.
        return pose, label
    
    def get_binned_pose(self, pose_label):
        bins = torch.tensor(range(-99, 102, 3))
        return torch.LongTensor(torch.bucketize(pose_label, bins))
    
    def get_pose_loss(self, x, c):
        """
            Args:
            1) x: Input generated images. (N,C,H,W). [-1,1]
            2) c: Rendering pose. (N,3). radians
            
            Return:
            CrossEntropy + MSE Loss
        """
        device = x.device
        idx_tensor = self.idx_tensor.to(device)
        
        # yaw_predicted, pitch_predicted, roll_predicted = self.forward(x) # [yaw, pitch, roll]. degree
        yaw_predicted, pitch_predicted, roll_predicted = self.hopenet(x) # [yaw, pitch, roll]. degree
        pose_label = c * 180/math.pi # radians -> degree
        pose_label_binned = get_binned_pose(pose_label) # (N,3)
        
        # CrossEntropy Loss
        loss = self.CrossEntropy(torch.stack([]))
        
        loss_yaw = self.CrossEntropy(yaw_predicted, pose_label_binned[...,0])
        loss_pitch = self.CrossEntropy(pitch_predicted, pose_label_binned[...,1])
        loss_roll = self.CrossEntropy(roll_predicted, pose_label_binned[...,2])
        
        loss = loss_yaw + loss_pitch + loss_roll
        
        # MSE Loss
        yaw_predicted = (torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99)
        pitch_predicted = (torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99)
        roll_predicted = (torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99)
        
        
        loss_yaw += self.alpha * self.MSE(yaw_predicted, pose_label[...,0])
        self.MSE(pitch_predicted, pose_label[...,1])
        self.MSE(roll_predicted, pose_label[...,2])
        
        

#----------------------------------------------------------------------------------        
"""
    Codes from the Github
    https://github.com/natanielruiz/deep-head-pose
"""
class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_angles(x)
        return x

class AlexNet(nn.Module):
    # AlexNet laid out as a Hopenet - classify Euler angles in bins and
    # regress the expected value.
    def __init__(self, num_bins):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc_yaw = nn.Linear(4096, num_bins)
        self.fc_pitch = nn.Linear(4096, num_bins)
        self.fc_roll = nn.Linear(4096, num_bins)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return yaw, pitch, roll
