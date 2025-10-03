from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from pprint import pprint
import torch

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ['bottle', 'bottle', 'diningtable', 'chair', 'chair', 'chair', 'pottedplant']

    def __getitem__(self, item):
        image, data = super().__getitem__(item)
        all_bboxes = []
        all_labels = []
        for obj in data["annotation"]["object"]:
            ymin = int(obj["bndbox"]["ymin"])
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])

            all_bboxes.append([xmin, ymin, xmax, ymax])
            all_labels.append(self.categories.index(obj["name"]))
        all_bboxes = torch.FloatTensor(all_bboxes)
        all_labels = torch.LongTensor(all_labels)
        # print(all_bboxes.shape)
        # print(all_labels)
        target = {
            "boxes" : all_bboxes,
            "labels" : all_labels
        }
        return image, data

if __name__ == '__main__':
    transform = ToTensor()
    dataset = VOCDataset(root="my_pascal_voc", year="2012", image_set='train', download=False, transform=transform)
    image, target = dataset[2000]
    # pprint(target["annotation"]["object"])
    # image.show()


    print(image.shape)
    print(target)