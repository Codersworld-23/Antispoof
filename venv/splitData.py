import os
import random
import shutil
from itertools import islice    

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
splitRatio = {"train":0.7, "val":0.2, "test":0.1 }
classes = ["fake", "real"]
try:
    shutil.rmtree(outputFolderPath)
    print("removed dir")
except OSError as e:
    os.mkdir(outputFolderPath)
    
    
#  DIR TO CREATE
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)


# GET NAMES
listNames = os.listdir(inputFolderPath)
print("length of listNames = ",len(listNames))
# print(listNames)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))
print("length of uniqueNames = ",len(uniqueNames))
# print(uniqueNames)


#  SHUFFLE
random.shuffle(uniqueNames)


# FIND NO OF IMAGES FOR EACH FOLDER
lenData = len(uniqueNames)
# print(f"total images {lenData}")
lenTrain = (int)(lenData * splitRatio["train"])
lenVal = (int)(lenData * splitRatio["val"])
lenTest = (int)(lenData * splitRatio["test"])
# print(f"total images: {lenData}, Split: {lenTrain} {lenVal} {lenTest}")



#  PUT REMAINING IMAGES IN TRAINING
if lenTrain + lenVal + lenTest != lenData:
    remaining = (lenData - (lenTrain + lenVal + lenTest))
    lenTrain += remaining    
# print(f"total images: {lenData}, Split: {lenTrain} {lenVal} {lenTest}")



# SPLIT THE LIST
lengthToSPLIT = [ lenTrain, lenVal, lenTest ]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSPLIT]
print(f"total images: {lenData}, \n Split: {len(Output[0])} {len(Output[1])} {len(Output[2])}")



# COPY THE FILES
sequence = ["train", "val", "test"]
for i,out in enumerate(Output):
    for filename in out:
        shutil.copy(f"{inputFolderPath}/{filename}.jpg", f"{outputFolderPath}/{sequence[i]}/images/{filename}.jpg")
        shutil.copy(f"{inputFolderPath}/{filename}.txt", f"{outputFolderPath}/{sequence[i]}/labels/{filename}.txt")



#  CREATING DATA.YAML FILE
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc : {len(classes)}\n\
names : {classes}'
    


f = open(f"{outputFolderPath}/data.yaml",'a')
f.write(dataYaml)
f.close()
