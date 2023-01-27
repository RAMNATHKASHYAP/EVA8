I Have used Two Models :
....................................

1st Model :

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 10, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)
        
................................................

2nd Model :
    class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(32, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x,dim=-1)
   ........................................................
   
   Comparison between two Models & Analysis:
   
 In the second model, the number of filters in each convolutional layer is reduced from 32 and 64 to 16 and 32 respectively. This results in a significant reduction in the number of parameters in the model.
 
 The second model uses batch normalization which can help to regularize the model and make it less prone to overfitting, and dropout which can also help to regularize the model.
 
 Therefore even after dropping of parameters from Accuracy is as par with 1st Model even better .
 
 1st Model Parameters:
 
 
15s
!pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: torchsummary in /usr/local/lib/python3.8/dist-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 28, 28]             320
            Conv2d-2           [-1, 64, 28, 28]          18,496
         MaxPool2d-3           [-1, 64, 14, 14]               0
            Conv2d-4          [-1, 128, 14, 14]          73,856
            Conv2d-5          [-1, 256, 14, 14]         295,168
         MaxPool2d-6            [-1, 256, 7, 7]               0
            Conv2d-7            [-1, 512, 5, 5]       1,180,160
            Conv2d-8           [-1, 1024, 3, 3]       4,719,616
            Conv2d-9             [-1, 10, 1, 1]          92,170
================================================================
Total params: 6,379,786
Trainable params: 6,379,786
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.51
Params size (MB): 24.34
Estimated Total Size (MB): 25.85
----------------------------------------------------------------
<ipython-input-2-98102ba2721d>:20: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
  
  
  2nd Model Parameeters :
  
  Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: torchsummary in /usr/local/lib/python3.8/dist-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
            Conv2d-3           [-1, 16, 28, 28]           2,320
       BatchNorm2d-4           [-1, 16, 28, 28]              32
         MaxPool2d-5           [-1, 16, 14, 14]               0
            Conv2d-6           [-1, 32, 14, 14]           4,640
       BatchNorm2d-7           [-1, 32, 14, 14]              64
            Conv2d-8           [-1, 32, 14, 14]           9,248
       BatchNorm2d-9           [-1, 32, 14, 14]              64
        MaxPool2d-10             [-1, 32, 7, 7]               0
           Conv2d-11             [-1, 64, 5, 5]          18,496
      BatchNorm2d-12             [-1, 64, 5, 5]             128
           Conv2d-13             [-1, 64, 3, 3]          36,928
      BatchNorm2d-14             [-1, 64, 3, 3]             128
AdaptiveAvgPool2d-15             [-1, 64, 1, 1]               0
          Dropout-16                   [-1, 64]               0
           Linear-17                   [-1, 10]             650
================================================================
Total params: 72,890
Trainable params: 72,890
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.64
Params size (MB): 0.28
Estimated Total Size (MB): 0.93
--------------------------------------------
  
  1st Model Accuracy :
   0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-8-98102ba2721d>:20: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
loss=1.9936423301696777 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.97it/s]

Test set: Average loss: 1.9692, Accuracy: 2803/10000 (28%)

loss=1.0576177835464478 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.83it/s]

Test set: Average loss: 1.0002, Accuracy: 6712/10000 (67%)

loss=0.7650046348571777 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.55it/s]

Test set: Average loss: 0.7522, Accuracy: 7761/10000 (78%)

loss=0.767174482345581 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.54it/s]

Test set: Average loss: 0.7429, Accuracy: 7798/10000 (78%)

loss=0.5503913760185242 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.23it/s]

Test set: Average loss: 0.5140, Accuracy: 7821/10000 (78%)

loss=0.4587317705154419 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.94it/s]

Test set: Average loss: 0.5047, Accuracy: 7850/10000 (78%)

loss=0.32255521416664124 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.18it/s]

Test set: Average loss: 0.2448, Accuracy: 8973/10000 (90%)

loss=0.19943810999393463 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.00it/s]

Test set: Average loss: 0.2448, Accuracy: 8969/10000 (90%)

loss=0.19925075769424438 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.47it/s]

Test set: Average loss: 0.2459, Accuracy: 8982/10000 (90%)

loss=0.14443139731884003 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.42it/s]

Test set: Average loss: 0.2483, Accuracy: 8978/10000 (90%)

loss=0.2881789207458496 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.88it/s]

Test set: Average loss: 0.2470, Accuracy: 8978/10000 (90%)

loss=0.4090680778026581 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.07it/s]

Test set: Average loss: 0.2492, Accuracy: 8971/10000 (90%)

loss=0.21926575899124146 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.05it/s]

Test set: Average loss: 0.2448, Accuracy: 8978/10000 (90%)

loss=0.12011966854333878 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.81it/s]

Test set: Average loss: 0.2455, Accuracy: 8990/10000 (90%)

loss=0.14407530426979065 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.94it/s]

Test set: Average loss: 0.2507, Accuracy: 8979/10000 (90%)

loss=0.26423758268356323 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.79it/s]

Test set: Average loss: 0.2443, Accuracy: 8994/10000 (90%)

loss=0.09622751921415329 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.09it/s]

Test set: Average loss: 0.2464, Accuracy: 8983/10000 (90%)

loss=0.09594421833753586 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.40it/s]

Test set: Average loss: 0.2492, Accuracy: 8980/10000 (90%)

loss=0.19201765954494476 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.28it/s]

Test set: Average loss: 0.2461, Accuracy: 8987/10000 (90%)
     
  
  2nd Model Accuracy:
  
  loss=0.11026168614625931 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.06it/s]

Test set: Average loss: 0.0448, Accuracy: 9872/10000 (99%)

loss=0.029440967366099358 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.44it/s]

Test set: Average loss: 0.0372, Accuracy: 9888/10000 (99%)

loss=0.041211239993572235 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.66it/s]

Test set: Average loss: 0.0264, Accuracy: 9920/10000 (99%)

loss=0.03548504039645195 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.36it/s]

Test set: Average loss: 0.0234, Accuracy: 9925/10000 (99%)

loss=0.04190903902053833 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.25it/s]

Test set: Average loss: 0.0223, Accuracy: 9929/10000 (99%)

loss=0.021777784451842308 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.70it/s]

Test set: Average loss: 0.0195, Accuracy: 9944/10000 (99%)

loss=0.01987290196120739 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.04it/s]

Test set: Average loss: 0.0299, Accuracy: 9905/10000 (99%)

loss=0.004199202638119459 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.59it/s]

Test set: Average loss: 0.0199, Accuracy: 9938/10000 (99%)

loss=0.013330187648534775 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.18it/s]

Test set: Average loss: 0.0157, Accuracy: 9951/10000 (100%)

loss=0.022823775187134743 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.00it/s]

Test set: Average loss: 0.0205, Accuracy: 9939/10000 (99%)

loss=0.01571989431977272 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.23it/s]

Test set: Average loss: 0.0194, Accuracy: 9940/10000 (99%)

loss=0.007272023241966963 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.37it/s]

Test set: Average loss: 0.0170, Accuracy: 9942/10000 (99%)

loss=0.01221280824393034 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.18it/s]

Test set: Average loss: 0.0191, Accuracy: 9942/10000 (99%)

loss=0.015282913111150265 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.27it/s]

Test set: Average loss: 0.0177, Accuracy: 9941/10000 (99%)

loss=0.008562359027564526 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.38it/s]

Test set: Average loss: 0.0170, Accuracy: 9950/10000 (100%)

loss=0.026490474119782448 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.20it/s]

Test set: Average loss: 0.0186, Accuracy: 9944/10000 (99%)

loss=0.0051473998464643955 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.43it/s]

Test set: Average loss: 0.0166, Accuracy: 9950/10000 (100%)

loss=0.012437216006219387 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.28it/s]

Test set: Average loss: 0.0193, Accuracy: 9934/10000 (99%)

loss=0.006074037868529558 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.25it/s]

Test set: Average loss: 0.0152, Accuracy: 9959/10000 (100%)
  
  
                                                                               
                                                                               
  REST EVERYTHING IS SAME IN BOTH OF THE MODELS
