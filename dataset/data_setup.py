from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

# t.compsoe is applied to all images
transform = T.Compose([
    T.Resize((224, 224)),  # Tiny ImageNet images are $64*  64$ pixels,this architecture recieve 24 as standard input
    T.ToTensor(),   #Converts the image data into a PyTorch Tensor.
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #adjusts the image's colors/intensities for scaled GD.
])

# root/{classX}/x001.jpg
#define dataset
tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)


#define dataloader
train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)