import albumentations as A
from torchvision import transforms as T


HV_FLIP = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
])

TO_TENSOR = T.ToTensor()


def train_transform(img, exp):
    return TO_TENSOR(HV_FLIP(image=img)['image'])


def dev_transform(img, exp):
    return TO_TENSOR(img)
