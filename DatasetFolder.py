from PIL import Image
from math import floor
import pandas as pd
import os

# Specify the image, annotations and destination path
imPath = './Images'
anPath = 'Annotations.csv'
dsPath = '../IndustryBiscuit_Folders'

# Set the dataset parameters
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

total_images = 4900
total_ok_images = 474 * 4
total_nok_images = total_images - total_ok_images

# Split the dataset based on the given ratios
nTrain_ok = floor(total_ok_images * train_ratio)
nValid_ok = floor(total_ok_images * valid_ratio)
nTest_ok = total_ok_images - nTrain_ok - nValid_ok

nTrain_nok = floor(total_nok_images * train_ratio)
nValid_nok = floor(total_nok_images * valid_ratio)
nTest_nok = total_nok_images - nTrain_nok - nValid_nok

# Defect ratios
rNComplete = 0.4
rSObject = 0.3
rCDefect = 0.3

# Counters initialization
cTrain_ok = 0
cValid_ok = 0
cTest_ok = 0

cTrainNC_nok = 0
cValidNC_nok = 0
cTestNC_nok = 0

cTrainSO_nok = 0
cValidSO_nok = 0
cTestSO_nok = 0

cTrainCD_nok = 0
cValidCD_nok = 0
cTestCD_nok = 0

# Defect limits
nTrNC = floor(nTrain_nok * rNComplete)
nVaNC = floor(nValid_nok * rNComplete)
nTeNC = floor(nTest_nok * rNComplete)

nTrSO = floor(nTrain_nok * rSObject)
nVaSO = floor(nValid_nok * rSObject)
nTeSO = floor(nTest_nok * rSObject)

nTrCD = floor(nTrain_nok * rCDefect)
nVaCD = floor(nValid_nok * rCDefect)
nTeCD = floor(nTest_nok * rCDefect)

# Create the directories for the image storage
if not os.path.exists(dsPath):

    # Create the folder structure
    os.mkdir(dsPath)
    os.mkdir(dsPath + '/train')
    os.mkdir(dsPath + '/train' + '/ok')
    os.mkdir(dsPath + '/train' + '/nok')
    os.mkdir(dsPath + '/valid')
    os.mkdir(dsPath + '/valid' + '/ok')
    os.mkdir(dsPath + '/valid' + '/nok')
    os.mkdir(dsPath + '/test')
    os.mkdir(dsPath + '/test' + '/ok')
    os.mkdir(dsPath + '/test' + '/nok')

    # Load the filenames and the annotation from the .csv file
    data = pd.read_csv(anPath, usecols=['file', 'classDescription'])

    augm = 1226

    for key in range(1, 1226):

        for temp in range(0, 4):

            if temp == 0:
                index = key
            else:
                index = augm
                augm += 1

            value = data.iloc[index - 1, :]

            # Open the image file
            im = Image.open(os.path.join(imPath, value[0]))

            # Split the images to the categories
            if value[1] == "Defect_No":
                if (cTrain_ok < nTrain_ok):
                    im.save(os.path.join(dsPath + '/train' + '/ok', value[0]), format='jpeg')
                    cTrain_ok += 1
                elif (cValid_ok < nValid_ok):
                    im.save(os.path.join(dsPath + '/valid' + '/ok', value[0]), format='jpeg')
                    cValid_ok += 1
                elif (cTest_ok < nTest_ok):
                    im.save(os.path.join(dsPath + '/test' + '/ok', value[0]), format='jpeg')
                    cTest_ok += 1

            elif value[1] == "Defect_Shape":
                if (cTrainNC_nok < nTrNC):
                    im.save(os.path.join(dsPath + '/train' + '/nok', value[0]), format='jpeg')
                    cTrainNC_nok += 1
                elif (cValidNC_nok < nVaNC):
                    im.save(os.path.join(dsPath + '/valid' + '/nok', value[0]), format='jpeg')
                    cValidNC_nok += 1
                elif (cTestNC_nok < nTeNC):
                    im.save(os.path.join(dsPath + '/test' + '/nok', value[0]), format='jpeg')
                    cTestNC_nok += 1

            elif value[1] == "Defect_Object":
                if (cTrainSO_nok < nTrSO):
                    im.save(os.path.join(dsPath + '/train' + '/nok', value[0]), format='jpeg')
                    cTrainSO_nok += 1
                elif (cValidSO_nok < nVaSO):
                    im.save(os.path.join(dsPath + '/valid' + '/nok', value[0]), format='jpeg')
                    cValidSO_nok += 1
                elif (cTestSO_nok < nTeSO):
                    im.save(os.path.join(dsPath + '/test' + '/nok', value[0]), format='jpeg')
                    cTestSO_nok += 1

            elif value[1] == "Defect_Color":
                if (cTrainCD_nok < nTrCD):
                    im.save(os.path.join(dsPath + '/train' + '/nok', value[0]), format='jpeg')
                    cTrainCD_nok += 1
                elif (cValidCD_nok < nVaCD):
                    im.save(os.path.join(dsPath + '/valid' + '/nok', value[0]), format='jpeg')
                    cValidCD_nok += 1
                elif (cTestCD_nok < nTeCD):
                    im.save(os.path.join(dsPath + '/test' + '/nok', value[0]), format='jpeg')
                    cTestCD_nok += 1

    # Print dataset statistics
    print("Dataset statistics:")
    print(f"Total images: {total_images}")
    print(f"Training set: {nTrain_ok + nTrain_nok} images")
    print(f"  OK: {nTrain_ok}")
    print(f"  NOK: {nTrain_nok}")
    print(f"Validation set: {nValid_ok + nValid_nok} images")
    print(f"  OK: {nValid_ok}")
    print(f"  NOK: {nValid_nok}")
    print(f"Test set: {nTest_ok + nTest_nok} images")
    print(f"   OK: {nTest_ok}")
    print(f"  NOK: {nTest_nok}")
    print("Defect ratios:")
    print(f"  Not complete: {rNComplete}")
    print(f"  Strange object: {rSObject}")
    print(f"  Color defect: {rCDefect}")

    print("Dataset created successfully...")

else:
    print("Folder structure with the dataset already exists...")

