import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchvision
import torch.nn as nn
image = torch.zeros((1, 3, 800, 800)).float()
bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]])
labels = torch.LongTensor([6,8])
sub_sample=16


# create a dummy image and set the volatile to be False
dummy_img = torch.zeros((1, 3, 800, 800)).float()
#print(dummy_img)

# List all the layers of the VGG16
model = torchvision.models.vgg16(pretrained=True)
fe = list(model.features)
print(fe)

# pass the image through the layers and check where you are getting this size
req_features = []
k = dummy_img.clone()
for i in fe:
    k = i(k)
    if k.size()[2] < 800 // 16:
        break
    req_features.append(i)
    out_channels = k.size()[1]
print(len(req_features))
print(out_channels)

# convert this list into a sequential module
faster_rcnn_fe_extractor = nn.Sequential(*req_features)
out_map = faster_rcnn_fe_extractor(image)
print(out_map.size())