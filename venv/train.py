from ultralytics import YOLO

model = YOLO('l_version_1_300.pt')  # load a pretrained model (recommended for training)

def main():
    model.train(data='Dataset/SplitData/dataOffline.yaml',epochs =3)

if __name__ == "__main__":
    main()