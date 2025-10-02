from torchvision.datasets import VOCDetection
from pprint import pprint

def train():
  train_dataset = VOCDetection(root  ="my_pascal_voc", year = "2012", image_set = 'train', download = False)
  image , label = train_dataset[2000]
  pprint(label)
  image.show()


if __name__ == '__main__':
  train()
