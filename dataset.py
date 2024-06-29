from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, ToPILImage
import torchvision

class MNIST(Dataset):
    def __init__(self):
        super(MNIST, self).__init__()
        self.dataset = torchvision.datasets.MNIST(root = "", train = True, download = True)
        self.image_convert = torchvision.transforms.Compose([
            ToTensor(),
            Resize((64, 64), antialias=True)
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.image_convert(image), label

if __name__ == "__main__":
    my_dataset = MNIST()
    image, label = my_dataset[0]
    
    pil_image = ToPILImage()(image)
    # pil_image_resized = pil_image.resize((224, 224))

    pil_image.save("mnist_image.png")
    print(f"Image saved with label: {label}")