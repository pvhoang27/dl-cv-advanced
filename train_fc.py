from torchvision.datasets import VOCDetection


def train():
  train_dataset = VOCDetection(root  ="my_pascal_voc", year = "2012", image_set = 'train', download = True)
  image , label = train_dataset[1000]
  print(label)
  print(image)


if __name__ == '__main__':
  train()
