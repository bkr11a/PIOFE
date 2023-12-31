__author__ = "Brad Rice"
__version__ = 0.1

import os
import cv2
import numpy as np

from utils.OpticalFlowUtils import FlowReader

class MPISintelHandler:
    def __init__(self):
        self.flowReader = FlowReader()

    def __call__(self):
        pass

    def run(self, path, w=1024, h=436, flow = False, greyscale = True):
        dictionary = self.generateDataDictionary(path=path)
        data = self.buildDataSet(dictionary=dictionary, w=w, h=h, flow=flow, greyscale=greyscale)

        return data

    def dictToNumpy(self, dictionary):
        keys = [k for k in dictionary]

        arr = np.vstack([dictionary[keys[0]], dictionary[keys[1]]])
        for i in range(2, len(keys)):
            arr = np.vstack([arr, dictionary[keys[i]]])

        return arr

    def loadData(self, path):
        with open(path) as f:
            arr = np.load(f)

        return arr

    def saveData(self, path, arr):
        # Save the array!
        with open(path, 'wb') as f:
            np.save(f, arr)

    def convertToLuminance(self, img):
        lab = img.copy()
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)

        l = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        return l, a, b

    def generateDataDictionary(self, path): 
        # TODO: Verify correctness of implementation
        dataDict = dict({})
        for root, dirs, files in os.walk(path):
            fs = []
            # print(f"{root}; {dirs}; {files}")
            d = root.split('/')[-1]
            if len(d) > 0:
                for file in files:
                    if file.endswith('.png') or file.endswith('.flo'):
                        path = os.path.join(root, file)
                        fs.append(path)
                        fs = sorted(fs)
                if len(fs) > 0:
                    dataDict[d] = fs

        return dataDict

    def buildDataSet(self, dictionary, w=1024, h=436, flow = False, greyscale = False):
        keys = list(dictionary.keys())
        imgsDictionary = {}
        for key in keys:
            imageArray = dictionary[key]
            imgs = []
            for i, imgName in enumerate(imageArray):
                if i < len(imageArray) - 1:
                    end = '\r'
                else:
                    end = '\n'
                print(f"Reading ({i+1} / {len(imageArray)}): {imgName}" + 100*" ", end = end)
                # path = os.path.join(imagePath, imgName)
                dim = (w, h)
                if flow:
                    # u, v, flows = convertImageToUVFlow(img)
                    flows = self.flowReader.readFlow(imgName)
                    flows = cv2.resize(flows, dim)
                    imgs.append(flows)
                else:
                    if i < len(imageArray) - 1:
                        completed = False
                        firstFrame = cv2.imread(imageArray[i])
                        secondFrame = cv2.imread(imageArray[i+1])
                        if firstFrame is not None:
                            if secondFrame is not None:
                                completed = True
                        j = 0
                        while not completed or j > 5:
                            j+=1
                            try:
                                firstFrame = cv2.imread(imageArray[i])
                                secondFrame = cv2.imread(imageArray[i+1])
                                if firstFrame is not None:
                                    if secondFrame is not None:
                                        completed = True
                            except Exception as err:
                                pass
                            
                        # print(firstFrame.shape)
                        # print(secondFrame.shape)
                        if greyscale:
                            firstFrame, _, _ = self.convertToLuminance(firstFrame)
                            secondFrame, _, _ = self.convertToLuminance(secondFrame)
                            # img = cv2.resize(img, dim)
                        firstFrame = cv2.resize(firstFrame, dim)
                        secondFrame = cv2.resize(secondFrame, dim)
                        imgs.append(np.array([firstFrame, secondFrame]))
                    
            imgs = np.array(imgs)
            imgsDictionary[key] = imgs

        return imgsDictionary