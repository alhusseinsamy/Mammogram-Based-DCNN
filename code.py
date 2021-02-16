import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import os
import scipy.ndimage
import cv2
from PIL import Image
import PIL
import keras.backend as K


path = "AllDICOMs"
masks_path = "Allmasks"

RCCMasspath = "RCCmassmask2"
RCCMcpath = "RCCmcmask2"
RMLOMasspath = "RMLOmassmask2"
RMLOMcpath = "RMLOmcmask2"
LCCMasspath = "LCCmassmask2"
LCCMcpath = "LCCmcmask2"
LMLOMasspath = "LMLOmassmask2"
LMLOMcpath = "LMLOmcmask2"

lstFilesDCM = []  # create an empty list

lstMasks = []

filenames = []
filenamesRCC = []
filenamesRMLO = []
filenamesLCC = []
filenamesLMLO = []

lstRMLO = []
lstLMLO = []
lstRCC = []
lstLCC = []



lstRCCMass = []
lstRCCMc = []
lstRMLOMass = []
lstRMLOMc = []
lstLCCMass = []
lstLCCMc = []
lstLMLOMass = []
lstLMLOMc = []


idsRMLO = []
idsLMLO = []
idsRCC = []
idsLCC = []

lstRMLO1 = []
lstLMLO1 = []
lstRCC1 = []
lstLCC1 = []


cases = []






def get_2d_data(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
            if ".dcm" in filename.lower():
                # lstFilesDCM.append(os.path.join(dirName,filename))
                if("R_CC_") in filename:
                    lstRCC.append(os.path.join(dirName,filename))
                if("L_CC_") in filename:
                    lstLCC.append(os.path.join(dirName,filename))
                if("R_ML_") in filename:
                    lstRMLO.append(os.path.join(dirName,filename))
                if("L_ML_") in filename:
                    lstLMLO.append(os.path.join(dirName,filename))

def get_masks_RCC_mass(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
    		if ".jpg" in filename.lower():
    			lstRCCMass.append(os.path.join(dirName,filename))
def get_masks_RCC_mc(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
    		if ".jpg" in filename.lower():
    			lstRCCMc.append(os.path.join(dirName,filename))
def get_masks_RMLO_mass(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
    		if ".jpg" in filename.lower():
    			lstRMLOMass.append(os.path.join(dirName,filename))
def get_masks_RMLO_mc(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
    		if ".jpg" in filename.lower():
    			lstRMLOMc.append(os.path.join(dirName,filename))

def get_masks_LCC_mass(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
    		if ".jpg" in filename.lower():
    			lstLCCMass.append(os.path.join(dirName,filename))
def get_masks_LCC_mc(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
    		if ".jpg" in filename.lower():
    			lstLCCMc.append(os.path.join(dirName,filename))
def get_masks_LMLO_mass(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
    		if ".jpg" in filename.lower():
    			lstLMLOMass.append(os.path.join(dirName,filename))
def get_masks_LMLO_mc(path):
    for dirName, subdirList, fileList in os.walk(path):
    	for filename in fileList:
    		if ".jpg" in filename.lower():
    			lstLMLOMc.append(os.path.join(dirName,filename))

get_2d_data(path)
get_masks_RCC_mass(RCCMasspath)
get_masks_RCC_mc(RCCMcpath)
get_masks_RMLO_mass(RMLOMasspath)
get_masks_RMLO_mc(RMLOMcpath)
get_masks_LCC_mass(LCCMasspath)
get_masks_LCC_mc(LCCMcpath)
get_masks_LMLO_mass(LMLOMasspath)
get_masks_LMLO_mc(LMLOMcpath)

for z in lstRCC:
    s=z.replace("AllDICOMs/", "")
    ind = s.index("_")
    s=s[ind+1:]
    ind1 = s.index("_")
    s=s[0:ind1]
    # print(s)
	# s=int(s)
    idsRCC.append(s)

for z in lstLCC:
    s=z.replace("AllDICOMs/", "")
    ind = s.index("_")
    s=s[ind+1:]
    ind1 = s.index("_")
    s=s[0:ind1]
    # print(s)
	# s=int(s)
    idsLCC.append(s)

for z in lstRMLO:
    s=z.replace("AllDICOMs/", "")
    ind = s.index("_")
    s=s[ind+1:]
    ind1 = s.index("_")
    s=s[0:ind1]
    # print(s)
	# s=int(s)
    idsRMLO.append(s)

for z in lstLMLO:
    s=z.replace("AllDICOMs/", "")
    ind = s.index("_")
    s=s[ind+1:]
    ind1 = s.index("_")
    s=s[0:ind1]
    # print(s)
	# s=int(s)
    idsLMLO.append(s)


#processing images functions
def process_image_RCC(index):
    if(lstRCC1[index] == "000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    RefDs = pydicom.read_file(lstRCC1[index])
    p = RefDs.pixel_array.astype(np.float32)
    # print(p.shape)
    # image_histogram,bins = np.histogram(p.flatten(),256,normed=True)
    # cdf = image_histogram.cumsum() # cumulative distribution function
    # cdf = 255 * cdf / cdf[-1]
    # image_equalized = np.interp(p.flatten(), bins[:-1], cdf)
    # image_equalized = image_equalized.reshape(p.shape)
    # image_equalized = scipy.ndimage.zoom(image_equalized.astype(np.float), 0.25)
    # print(image_equalized.shape)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    # plt.imshow(res, cmap=plt.cm.bone)
    # plt.show()
    res *= (255.0/res.max())
    # res = (255.0 / res.max() * (res - res.min())).astype(np.float32)
    # res = (255.0 / res.max() * (res - res.min())).astype(np.uint8)
    # print(res.shape)
    # return res.astype(np.float32)*1./255
    return res.astype(np.float32)

def process_image_RMLO(index):
    if(lstRMLO1[index] == "000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    RefDs = pydicom.read_file(lstRMLO1[index])
    p = RefDs.pixel_array.astype(np.float32)
    # print(p.shape)
    # image_histogram,bins = np.histogram(p.flatten(),256,normed=True)
    # cdf = image_histogram.cumsum() # cumulative distribution function
    # cdf = 255 * cdf / cdf[-1]
    # image_equalized = np.interp(p.flatten(), bins[:-1], cdf)
    # image_equalized = image_equalized.reshape(p.shape)
    # image_equalized = scipy.ndimage.zoom(image_equalized.astype(np.float), 0.25)
    # print(image_equalized.shape)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    # plt.imshow(res, cmap=plt.cm.bone)
    # plt.show()
    res *= (255.0/res.max())
    # res = (255.0 / res.max() * (res - res.min())).astype(np.float32)
    # res = (255.0 / res.max() * (res - res.min())).astype(np.uint8)
    # print(res.shape)
    # return res.astype(np.float32)*1./255
    return res.astype(np.float32)

def process_image_LCC(index):
    if(lstLCC1[index] == "000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    RefDs = pydicom.read_file(lstLCC1[index])
    p = RefDs.pixel_array.astype(np.float32)
    # print(p.shape)
    # image_histogram,bins = np.histogram(p.flatten(),256,normed=True)
    # cdf = image_histogram.cumsum() # cumulative distribution function
    # cdf = 255 * cdf / cdf[-1]
    # image_equalized = np.interp(p.flatten(), bins[:-1], cdf)
    # image_equalized = image_equalized.reshape(p.shape)
    # image_equalized = scipy.ndimage.zoom(image_equalized.astype(np.float), 0.25)
    # print(image_equalized.shape)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    # plt.imshow(res, cmap=plt.cm.bone)
    # plt.show()
    res *= (255.0/res.max())
    # res = (255.0 / res.max() * (res - res.min())).astype(np.float32)
    # res = (255.0 / res.max() * (res - res.min())).astype(np.uint8)
    # print(res.shape)
    # return res.astype(np.float32)*1./255
    return res.astype(np.float32)


def process_image_LMLO(index):
    if(lstLMLO1[index] == "000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    RefDs = pydicom.read_file(lstLMLO1[index])
    p = RefDs.pixel_array.astype(np.float32)
    # print(p.shape)
    # image_histogram,bins = np.histogram(p.flatten(),256,normed=True)
    # cdf = image_histogram.cumsum() # cumulative distribution function
    # cdf = 255 * cdf / cdf[-1]
    # image_equalized = np.interp(p.flatten(), bins[:-1], cdf)
    # image_equalized = image_equalized.reshape(p.shape)
    # image_equalized = scipy.ndimage.zoom(image_equalized.astype(np.float), 0.25)
    # print(image_equalized.shape)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    # plt.imshow(res, cmap=plt.cm.bone)
    # plt.show()
    res *= (255.0/res.max())
    # res = (255.0 / res.max() * (res - res.min())).astype(np.float32)
    # res = (255.0 / res.max() * (res - res.min())).astype(np.uint8)
    # print(res.shape)
    # return res.astype(np.float32)*1./255
    return res.astype(np.float32)

# print(lstRCCMass[93])

def process_RCC_mass(index):
    # s=z.replace("AllDICOMs/", "")
    if (lstRCC1[index]=="000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    im = lstRCC1[index].replace("AllDICOMs/", "RCCmassmask2/")
    im = im.replace(".dcm", ".jpg")
    ind = lstRCCMass.index(im)
    # print(ind)
    # print(lstRCC1[index])
    # print(lstRCCMass[ind])
    image = Image.open(lstRCCMass[ind])
    p= np.array(image)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    return res.astype(np.float32)

def process_RCC_mc(index):
    # s=z.replace("AllDICOMs/", "")
    if (lstRCC1[index]=="000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    im = lstRCC1[index].replace("AllDICOMs/", "RCCmcmask2/")
    im = im.replace(".dcm", ".jpg")
    ind = lstRCCMc.index(im)
    # print(ind)
    # print(lstRCC1[index])
    # print(lstRCCMass[ind])
    image = Image.open(lstRCCMc[ind])
    p= np.array(image)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    return res.astype(np.float32)

def process_RMLO_mass(index):
    # s=z.replace("AllDICOMs/", "")
    if (lstRMLO1[index]=="000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    im = lstRMLO1[index].replace("AllDICOMs/", "RMLOmassmask2/")
    im = im.replace(".dcm", ".jpg")
    ind = lstRMLOMass.index(im)
    # print(ind)
    # print(lstRMLO1[index])
    # print(lstRMLOMass[ind])
    image = Image.open(lstRMLOMass[ind])
    p= np.array(image)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    return res.astype(np.float32)

def process_RMLO_mc(index):
    # s=z.replace("AllDICOMs/", "")
    if (lstRMLO1[index]=="000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    im = lstRMLO1[index].replace("AllDICOMs/", "RMLOmcmask2/")
    im = im.replace(".dcm", ".jpg")
    ind = lstRMLOMc.index(im)
    # print(ind)
    # print(lstRMLO1[index])
    # print(lstRMLOMass[ind])
    image = Image.open(lstRMLOMc[ind])
    p= np.array(image)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    return res.astype(np.float32)


#-----------------------------------

def process_LCC_mass(index):
    # s=z.replace("AllDICOMs/", "")
    if (lstLCC1[index]=="000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    im = lstLCC1[index].replace("AllDICOMs/", "LCCmassmask2/")
    im = im.replace(".dcm", ".jpg")
    ind = lstLCCMass.index(im)
    # print(ind)
    # print(lstLCC1[index])
    # print(lstLCCMass[ind])
    image = Image.open(lstLCCMass[ind])
    p= np.array(image)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    return res.astype(np.float32)

def process_LCC_mc(index):
    # s=z.replace("AllDICOMs/", "")
    if (lstLCC1[index]=="000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    im = lstLCC1[index].replace("AllDICOMs/", "LCCmcmask2/")
    im = im.replace(".dcm", ".jpg")
    ind = lstLCCMc.index(im)
    # print(ind)
    # print(lstLCC1[index])
    # print(lstLCCMass[ind])
    image = Image.open(lstLCCMc[ind])
    p= np.array(image)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    return res.astype(np.float32)

def process_LMLO_mass(index):
    # s=z.replace("AllDICOMs/", "")
    if (lstLMLO1[index]=="000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    im = lstLMLO1[index].replace("AllDICOMs/", "LMLOmassmask2/")
    im = im.replace(".dcm", ".jpg")
    ind = lstLMLOMass.index(im)
    # print(ind)
    # print(lstLMLO1[index])
    # print(lstLMLOMass[ind])
    image = Image.open(lstLMLOMass[ind])
    p= np.array(image)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    return res.astype(np.float32)

def process_LMLO_mc(index):
    # s=z.replace("AllDICOMs/", "")
    if (lstLMLO1[index]=="000000000000000"):
        return np.zeros([150, 150, 1], np.float32)
    im = lstLMLO1[index].replace("AllDICOMs/", "LMLOmcmask2/")
    im = im.replace(".dcm", ".jpg")
    ind = lstLMLOMc.index(im)
    # print(ind)
    # print(lstLMLO1[index])
    # print(lstLMLOMass[ind])
    image = Image.open(lstLMLOMc[ind])
    p= np.array(image)
    res = cv2.resize(p, dsize=(150, 150), interpolation=cv2.INTER_LANCZOS4)
    return res.astype(np.float32)



lstRCC.sort()
lstLCC.sort()
lstRMLO.sort()
lstLMLO.sort()

idsRCC.sort()
idsLCC.sort()
idsRMLO.sort()
idsLMLO.sort()

allpatientsids = ((set(idsRCC).union(set(idsLCC))).union(set(idsRMLO))).union(set(idsLMLO))
# print(len(allpatientsids))

fullcases = []
onlyright = []
onlyleft = []


for x in allpatientsids:
    if (x in idsRCC) and (x in idsLCC ) and (x in idsRMLO) and (x in idsLMLO):
        fullcases.append(x)


temp = allpatientsids - set(fullcases)

for x in temp:
    if (x in idsRCC) and (x in idsRMLO):
        onlyright.append(x)

temp1 = temp - set(onlyright)

for x in temp1:
    if (x in idsLCC) and (x in idsLMLO):
        onlyleft.append(x)

# print(len(allpatientsids))
# print(len(fullcases))
# print(len(onlyright))
# print(len(onlyleft))

# print(list(set(allpatientsids) - (set(onlyright).union(set(onlyleft))).union(set(fullcases)) ))

notneeded = (list(set(allpatientsids) - (set(onlyright).union(set(onlyleft))).union(set(fullcases)) ))
# print(idsRCC[0])



# print(notneeded)


# for i in range(0, len(idsRMLO)):
#     if(notneeded[0] == idsRMLO[i]) or (notneeded[1] == idsRMLO[i]):
#         print(idsRMLO[i])
#         print(i)
# print(len(idsRMLO))

idsRMLO.pop(1)
idsRMLO.pop(13)
# print(len(idsRMLO))






# for i in range(0, len(idsRCC)):
#     if(idsRCC[i] not in idsRMLO):
#         print(i)
#         print(idsRCC[i])
idsRCC.pop(21)
idsRCC.pop(21)
idsRCC.pop(34-1)

# print(idsRMLO)




# for i in range(0, len(idsRMLO)):
#     if(idsRMLO[i] not in idsRCC):
#         print(i)
#         print(idsRMLO[i])

# idsRMLO.pop(26)
# idsRMLO.pop(34)
# idsRMLO.pop(41-1)
# idsRMLO.pop(88-2)

idsRMLO.pop(26)
idsRMLO.pop(41)
idsRMLO.pop(87)

# print(idsLCC)




# for i in range(0, len(idsLCC)):
#     if(idsLCC[i] not in idsLMLO):
#         print(i)
idsLCC.pop(7)
idsLCC.pop(27)
idsLCC.pop(46-1)
idsLCC.pop(50-2)
idsLCC.pop(62-3)
#
# print(idsLMLO)

# for i in range(0, len(idsLMLO)):
#     if(idsLMLO[i] not in idsLCC):
#         print(i)
#         print(idsLMLO[i])



idsLMLO.pop(52)
idsLMLO.pop(76)
idsLMLO.pop(88)

# b=15
# print(idsRCC[b])
# print(idsRMLO[b])
# print(idsLCC[b])
# print(idsLMLO[b])

# idsRCC = list(filter(lambda a: a != "465aa5ec1b59efc6", idsRCC))
# idsRMLO = list(filter(lambda a: a != "465aa5ec1b59efc6", idsRMLO))
# idsLCC = list(filter(lambda a: a != "465aa5ec1b59efc6", idsLCC))
# idsLMLO = list(filter(lambda a: a != "465aa5ec1b59efc6", idsLMLO))

idsRMLO.remove("465aa5ec1b59efc6")
idsRCC.remove("cc9e66c5b31baab8")
idsRMLO.remove("de4c34099d6ef8de")

idsLCC.remove("465aa5ec1b59efc6")


# print(idsRCC[10])

# print(idsLCC[10])



counterr = 0

counterl=0

while(counterr < len(idsRCC)):
    if (idsRCC[counterr] not in idsLCC) and (idsRCC[counterr]!="000000000000000"):
        idsLCC.insert(counterl, "000000000000000")
        idsLMLO.insert(counterl, "000000000000000")
        counterl+=1
        counterr+=1


    elif (idsLCC[counterl] not in idsRCC) and (idsLCC[counterl]!="000000000000000"):
        idsRCC.insert(counterr, "000000000000000")
        idsRMLO.insert(counterr, "000000000000000")
        counterr+=1
        counterl+=1
    else:
        counterr+=1
        counterl+=1


# for i in range(0, len(idsRCC)):
#     if (idsRCC[i] not in idsLCC):
#         idsLCC.insert(i, "000000000000000")
#         i+=1
#         # idsLMLO.insert(i, "000000000000000")
#     if (idsLCC[i] not in idsRCC):
#         idsRCC.insert(i, "000000000000000")
#         i+=1
#         # idsRMLO.insert(i+1, "000000000000000")
#         i+=1
#         # print(i)


# for i in range(0, len(idsRCC)):
#     if (idsRCC[i] not in idsLCC):
#         idsLCC.insert(i, "000000000000000")

# for i in range(0, len(idsLCC)):
#     if(idsLCC[i] != "000000000000000"):
#         if (idsLCC[i] not in idsRCC):
#             idsRCC.insert(i, "000000000000000")


# idsLCC.insert(4, "000000000000000")
# # idsRCC.insert(5, "000000000000000")



# for i in range(0, len(idsRCC)):
#     print(str(i)+ " RCC: " + idsRCC[i] + " LCC: "+idsLCC[i])

# print(len(idsRCC))
# print(len(idsRMLO))
# print(len(idsLCC))
# print(len(idsLMLO))



# print(len(fullcases))


# print(idsRCC[45])
# print(idsRMLO[45])
# print(idsLCC[45])
# print(idsLMLO[45])

# print(len(idsRCC))
# print(len(lstRCC))

# print(idsRCC)

# for a, x in enumerate(idsRCC):
#     num = idsRCC.count(x)



taken = []

for i in range(0, len(idsRCC)):
    if(idsRCC[i] == "000000000000000"):
                lstRCC1.append("000000000000000")
    for j in range(0, len(lstRCC)):
        if j not in taken:
            if(idsRCC[i] in lstRCC[j]):
                # print("id: "+idsRCC[i]+ "path: " +lstRCC[j])
                lstRCC1.append(lstRCC[j])
                taken.append(j)
                break


taken = []
for i in range(0, len(idsLCC)):
    if(idsLCC[i] == "000000000000000"):
                lstLCC1.append("000000000000000")
    for j in range(0, len(lstLCC)):
        if j not in taken:
            if(idsLCC[i] in lstLCC[j]):
                # print("id: "+idsRCC[i]+ "path: " +lstRCC[j])
                lstLCC1.append(lstLCC[j])
                taken.append(j)
                break


taken = []
for i in range(0, len(idsLMLO)):
    if(idsLMLO[i] == "000000000000000"):
                lstLMLO1.append("000000000000000")
    for j in range(0, len(lstLMLO)):
        if j not in taken:
            if(idsLMLO[i] in lstLMLO[j]):
                # print("id: "+idsRCC[i]+ "path: " +lstRCC[j])
                lstLMLO1.append(lstLMLO[j])
                taken.append(j)
                break


taken = []
for i in range(0, len(idsRMLO)):
    if(idsRMLO[i] == "000000000000000"):
                lstRMLO1.append("000000000000000")
    for j in range(0, len(lstRMLO)):
        if j not in taken:
            if(idsRMLO[i] in lstRMLO[j]):
                # print("id: "+idsRCC[i]+ "path: " +lstRCC[j])
                lstRMLO1.append(lstRMLO[j])
                taken.append(j)
                break

# print(lstRCC1)
# print(lstLCC1[46])
# print(lstRMLO1[46])
# print(lstLMLO1[46])

# print(len(set(lstRCC1)))
# print(len(set(lstLCC1)))
# print(len(set(lstRMLO1)))
# print(len(set(lstLMLO1)))

for z in lstRCC1:
    if (z=="000000000000000"):
        filenamesRCC.append(z)
    else:
        s=z.replace("AllDICOMs/", "")
        ind = s.index("_")
        s=s[0:ind]
        s=int(s)
        filenamesRCC.append(s)

for z in lstLCC1:
    if (z=="000000000000000"):
        filenamesLCC.append(z)
    else:
        s=z.replace("AllDICOMs/", "")
        ind = s.index("_")
        s=s[0:ind]
        s=int(s)
        filenamesLCC.append(s)


for z in lstLMLO1:
    if (z=="000000000000000"):
        filenamesLMLO.append(z)
    else:
        s=z.replace("AllDICOMs/", "")
        ind = s.index("_")
        s=s[0:ind]
        s=int(s)
        filenamesLMLO.append(s)

for z in lstRMLO1:
    if (z=="000000000000000"):
        filenamesRMLO.append(z)
    else:
        s=z.replace("AllDICOMs/", "")
        ind = s.index("_")
        s=s[0:ind]
        s=int(s)
        filenamesRMLO.append(s)


# print(len(filenamesRCC))
# print(len(filenamesRMLO))
# print(len(filenamesLCC))
# print(len(filenamesLMLO))





labels_csv = pd.read_csv('INbreast.csv', delimiter=";", index_col="File Name")

train_labelsRCC, train_labelsLCC, train_labelsRMLO, train_labelsLMLO = labels_csv.loc[filenamesRCC, ["Bi-Rads"]], labels_csv.loc[filenamesLCC, ["Bi-Rads"]], labels_csv.loc[filenamesRMLO, ["Bi-Rads"]], labels_csv.loc[filenamesLMLO, ["Bi-Rads"]]

# train_labelsRCC['Bi-Rads'] = train_labelsRCC['Bi-Rads'].map({'4c': 1})
train_labelsRCC['Bi-Rads'].fillna(17, inplace=True)
train_labelsRMLO['Bi-Rads'].fillna(17, inplace=True)
train_labelsLCC['Bi-Rads'].fillna(17, inplace=True)
train_labelsLMLO['Bi-Rads'].fillna(17, inplace=True)






# k=43
# print(train_labelsRCC.iloc[[k]])
# print(train_labelsRMLO.iloc[[k]])
#
# print(train_labelsLCC.iloc[[k]])
# print(train_labelsLMLO.iloc[[k]])



# print(train_labelsRCC.get_value(2,0, takeable=True))


# train_labelsRCC.iloc[[0]] = 15
train_labels = pd.DataFrame().reindex_like(train_labelsRCC)
#
#
#
#
#
# for i in range(0, len(train_labelsRCC)):
#     train_labels.iloc[[i]] = max(str(train_labelsRCC.get_value(i, 0, takeable=True)), str(train_labelsRMLO.get_value(i, 0, takeable=True)), str(train_labelsLCC.get_value(i, 0, takeable=True)), str(train_labelsLMLO.get_value(i, 0, takeable=True)))
#
#
#
# train_labelsRCC['Bi-Rads'] = train_labelsRCC['Bi-Rads'].map({'1': 0, '2':0, '3': 0, '4a':1 , '4b':1 , '4c':1  , '5':1 , '6':1 })
# train_labelsRMLO['Bi-Rads'] = train_labelsRMLO['Bi-Rads'].map({'1': 0, '2':0, '3': 0, '4a':1 , '4b':1 , '4c':1  , '5':1 , '6':1 })
# train_labelsLCC['Bi-Rads'] = train_labelsLCC['Bi-Rads'].map({'1': 0, '2':0, '3': 0, '4a':1 , '4b':1 , '4c':1  , '5':1 , '6':1 })
# train_labelsLMLO['Bi-Rads'] = train_labelsLMLO['Bi-Rads'].map({'1': 0, '2':0, '3': 0, '4a':1 , '4b':1 , '4c':1  , '5':1 , '6':1 })
#
# train_labels=train_labels.astype(np.uint8).as_matrix()

# print(train_labelsRCC)


for i in range(0, len(train_labelsRCC)):
    if(train_labelsRCC.get_value(i, 0, takeable=True) == 17):
        train_labelsRCC.iloc[[i]] = train_labelsLCC.get_value(i, 0, takeable=True)

    if(train_labelsRMLO.get_value(i, 0, takeable=True) == 17):
        train_labelsRMLO.iloc[[i]] = train_labelsLMLO.get_value(i, 0, takeable=True)

    if(train_labelsLCC.get_value(i, 0, takeable=True) == 17):
        train_labelsLCC.iloc[[i]] = train_labelsRCC.get_value(i, 0, takeable=True)

    if(train_labelsLMLO.get_value(i, 0, takeable=True) == 17):
        train_labelsLMLO.iloc[[i]] = train_labelsRMLO.get_value(i, 0, takeable=True)



train_labels = (train_labelsRCC.append(train_labelsLCC))

train_labels1 = (train_labelsRMLO.append(train_labelsLMLO))

for i in range(0, len(train_labels)):
    if(train_labels.get_value(i, 0, takeable=True) < train_labels1.get_value(i, 0, takeable=True)):
        train_labels.iloc[[i]] = train_labels1.get_value(i, 0, takeable=True)



train_labels['Bi-Rads'] = train_labels['Bi-Rads'].map({'1': 0, '2':0, '3': 0, '4a':1 , '4b':1 , '4c':1  , '5':1 , '6':1 })


train_labels=train_labels.astype(np.uint8).as_matrix()

print(train_labels)

# for u in range(0, len(train_labels)):
#     print(train_labels.iloc[[u]])
#     print(train_labels1.iloc[[u]])
#     print("______________________________________________________________________________________________")





# d=3
# # print(lstRCC1[d])
# print(train_labelsRCC.iloc[[d]])
# print(train_labelsRMLO.iloc[[d]])
# print(train_labelsLCC.iloc[[d]])
# print(train_labelsLMLO.iloc[[d]])
# # print(train_labels.iloc[[d]])
# # print(train_labels[d])



train_features_RCC = np.zeros([len(filenamesRCC), 150, 150, 1], np.float32)
train_features_RMLO = np.zeros([len(filenamesRMLO), 150, 150, 1], np.float32)
train_features_LCC = np.zeros([len(filenamesLCC), 150, 150, 1], np.float32)
train_features_LMLO = np.zeros([len(filenamesLMLO), 150, 150, 1], np.float32)

mass_features_RCC = np.zeros([len(filenamesRCC), 150, 150, 1], np.float32)
mass_features_RMLO = np.zeros([len(filenamesRMLO), 150, 150, 1], np.float32)
mass_features_LCC = np.zeros([len(filenamesLCC), 150, 150, 1], np.float32)
mass_features_LMLO = np.zeros([len(filenamesLMLO), 150, 150, 1], np.float32)

mc_features_RCC = np.zeros([len(filenamesRCC), 150, 150, 1], np.float32)
mc_features_RMLO = np.zeros([len(filenamesRMLO), 150, 150, 1], np.float32)
mc_features_LCC = np.zeros([len(filenamesLCC), 150, 150, 1], np.float32)
mc_features_LMLO = np.zeros([len(filenamesLMLO), 150, 150, 1], np.float32)

# process_RMLO_mc(0)

for i in range(len(filenamesRCC)):
    RCCview = process_image_RCC(i)
    RMLOview = process_image_RMLO(i)
    LCCview = process_image_LCC(i)
    LMLOview = process_image_LMLO(i)

    RCCviewmass = process_RCC_mass(i)
    RMLOviewmass = process_RMLO_mass(i)
    LCCviewmass = process_LCC_mass(i)
    LMLOviewmass = process_LMLO_mass(i)

    RCCviewmc = process_RCC_mc(i)
    RMLOviewmc = process_RMLO_mc(i)
    LCCviewmc = process_LCC_mc(i)
    LMLOviewmc = process_LMLO_mc(i)
#-----------------------------------------------------------------
    RCCview = RCCview.reshape(150, 150, 1)
    RMLOview = RMLOview.reshape(150, 150, 1)
    LCCview = LCCview.reshape(150, 150, 1)
    LMLOview = LMLOview.reshape(150, 150, 1)

    RCCviewmass = RCCviewmass.reshape(150, 150, 1)
    RMLOviewmass = RMLOviewmass.reshape(150, 150, 1)
    LCCviewmass = LCCviewmass.reshape(150, 150, 1)
    LMLOviewmass = LMLOviewmass.reshape(150, 150, 1)

    RCCviewmc = RCCviewmc.reshape(150, 150, 1)
    RMLOviewmc = RMLOviewmc.reshape(150, 150, 1)
    LCCviewmc = LCCviewmc.reshape(150, 150, 1)
    LMLOviewmc = LMLOviewmc.reshape(150, 150, 1)
#------------------------------------------------------------------

    train_features_RCC[i] = RCCview
    train_features_RMLO[i] = RMLOview
    train_features_LCC[i] = LCCview
    train_features_LMLO[i] = LMLOview

    mass_features_RCC[i] = RCCviewmass
    mass_features_RMLO[i] = RMLOviewmass
    mass_features_LCC[i] = LCCviewmass
    mass_features_LMLO[i] = LMLOviewmass

    mc_features_RCC[i] = RCCviewmc
    mc_features_RMLO[i] = RMLOviewmc
    mc_features_LCC[i] = LCCviewmc
    mc_features_LMLO[i] = LMLOviewmc
#--------------------------------------------------------------------
# p = np.random.permutation(len(train_features))
# train_features =  train_features[p]
# mask_features_mass =  mask_features_mass[p]
# mask_features_mc = mask_features_mc[p]

train_features_CC = np.append(train_features_RCC, train_features_LCC, axis=0)
train_features_MLO = np.append(train_features_RMLO, train_features_LMLO, axis=0)

mass_features_CC = np.append(mass_features_RCC, mass_features_LCC, axis=0)
mass_features_MLO = np.append(mass_features_RMLO, mass_features_LMLO, axis=0)

mc_features_CC = np.append(mc_features_RCC, mc_features_LCC, axis=0)
mc_features_MLO = np.append(mc_features_RMLO, mc_features_LMLO, axis=0)



p = np.random.permutation(len(train_features_CC))
train_features_CC =  train_features_CC[p]
train_features_MLO =  train_features_MLO[p]


mass_features_CC =  mass_features_CC[p]
mass_features_MLO =  mass_features_MLO[p]


mc_features_CC =  mc_features_CC[p]
mc_features_MLO =  mc_features_MLO[p]

train_labels = train_labels[p]


# print(len(train_features_CC))

# print(len(train_features_CC))
# print(len(train_features_RCC))
# print(len(train_features_LCC))
#
#
def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall



def generate_generator_multiple(generator1, generator2, generator1mass, generator2mass, generator1mc, generator2mc, batch_size, train_features_CC, train_features_MLO,
mass_features_CC, mass_features_MLO,
mc_features_CC, mc_features_MLO, train_labels):
    generator1.fit(train_features_CC)
    genX1 = generator1.flow(train_features_CC,
                        train_labels,
                        batch_size = batch_size,
                        shuffle=False)
    generator2.fit(train_features_MLO)
    genX2 = generator2.flow(train_features_MLO,
                        train_labels,
                        batch_size = batch_size,
                        shuffle=False)

#--------------------------------------------------------------------------------------------------
    generator1mass.fit(mass_features_CC)
    genX1mass = generator1mass.flow(mass_features_CC,
                        train_labels,
                        batch_size = batch_size,
                        shuffle=False)
    generator2mass.fit(mass_features_MLO)
    genX2mass = generator2mass.flow(mass_features_MLO,
                        train_labels,
                        batch_size = batch_size,
                        shuffle=False)

#--------------------------------------------------------------------------------------------------------------------
    generator1mc.fit(mc_features_CC)
    genX1mc = generator1mc.flow(mc_features_CC,
                        train_labels,
                        batch_size = batch_size,
                        shuffle=False)
    generator2mc.fit(mc_features_MLO)
    genX2mc = generator2mc.flow(mc_features_MLO,
                        train_labels,
                        batch_size = batch_size,
                        shuffle=False)


    while True:
            X1i = genX1.next()
            X2i = genX2.next()


            X1imass = genX1mass.next()
            X2imass = genX2mass.next()


            X1imc = genX1mc.next()
            X2imc = genX2mc.next()

            yield [X1i[0], X1imass[0], X1imc[0], X2i[0], X2imass[0], X2imc[0]], X2i[1]  #Yield both images and their mutual label


#-----------------------------------------------------------------------------------------------------------------------------------------------
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import SGD
from keras.models import Sequential, Model, Input
from keras.layers import Dropout, Flatten, Dense, Input, Merge
from sklearn.model_selection import StratifiedKFold
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
#----------------------------------------------------Train top model first---------------------------------------------------------------------


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

skf = StratifiedKFold(n_splits=5) #random_state=7
cvscores = []
count = 1

# model_weights_path ="thedata.h5"
# masks_weights_path ="masks.h5"

for (train_index, test_index) in skf.split(train_features_CC, train_labels):
    # print(train_features[train_index])
    train_datagen_CC = ImageDataGenerator(rescale = 1./255,
                                       # shear_range = 0.2,
                                       # zoom_range = 0.2,
                                       # horizontal_flip = True
                                       )

    test_datagen_CC = ImageDataGenerator(rescale = 1./255,
                                        # horizontal_flip = True
                                        )
#---------------------------------------------------
    train_datagen_MLO = ImageDataGenerator(rescale = 1./255,
                                       # shear_range = 0.2,
                                       # zoom_range = 0.2,
                                       # horizontal_flip = True
                                       )

    test_datagen_MLO = ImageDataGenerator(rescale = 1./255,
                                        # horizontal_flip = True
                                        )

#----------------------------------------------------------------------------





    train_datagen_CC_mass = ImageDataGenerator(rescale = 1./255,
                                       # shear_range = 0.2,
                                       # zoom_range = 0.2,
                                       # horizontal_flip = True
                                       )

    test_datagen_CC_mass = ImageDataGenerator(rescale = 1./255,
                                                # horizontal_flip = True
                                                )
#---------------------------------------------------
    train_datagen_MLO_mass = ImageDataGenerator(rescale = 1./255,
                                       # shear_range = 0.2,
                                       # zoom_range = 0.2,
                                       # horizontal_flip = True
                                       )

    test_datagen_MLO_mass = ImageDataGenerator(rescale = 1./255,
                                                # horizontal_flip = True
                                                )

#----------------------------------------------------------------------------




    train_datagen_CC_mc = ImageDataGenerator(rescale = 1./255,
                                       # shear_range = 0.2,
                                       # zoom_range = 0.2,
                                       # horizontal_flip = True
                                       )

    test_datagen_CC_mc = ImageDataGenerator(rescale = 1./255,
                                            # horizontal_flip = True
                                            )
#---------------------------------------------------
    train_datagen_MLO_mc = ImageDataGenerator(rescale = 1./255,
                                       # shear_range = 0.2,
                                       # zoom_range = 0.2,
                                       # horizontal_flip = True
                                       )

    test_datagen_MLO_mc = ImageDataGenerator(rescale = 1./255,
                                            # horizontal_flip = True
                                            )




    print ("Running Fold", count, "/", 5)
    count+=1
    model = None

    #Model1
    inp1 = Input(shape=(150, 150, 1))
    conv1 = Convolution2D(64, (11, 11), activation='relu', strides=4)(inp1)
    batchnorm1 = BatchNormalization()(conv1)
    maxp1 = MaxPooling2D((2, 2))(batchnorm1)

    zeropad = ZeroPadding2D((2,2))(maxp1)
    conv2 = Convolution2D(256, (5, 5), activation='relu', strides=1)(zeropad)
    batchnorm2 = BatchNormalization()(conv2)
    maxp2 = MaxPooling2D((2, 2))(batchnorm2)

    zeropad1 = ZeroPadding2D((1,1))(maxp2)
    conv3 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad1)

    zeropad2 = ZeroPadding2D((1,1))(conv3)
    conv4 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad2)

    zeropad3 = ZeroPadding2D((1,1))(conv4)
    conv5 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad3)
    maxp3 = MaxPooling2D((2, 2))(conv5)

    flt1 = Flatten()(maxp3)
    dn1 = Dense(1024, activation='relu')(flt1)
    drp1 = Dropout(0.5)(dn1)

    dn2 = Dense(1024, activation='relu')(drp1)
    drp2 = Dropout(0.5)(dn2)

    op1 = Dense(1, activation='sigmoid')(drp2)

    pre_model = Model(input=inp1, output=op1)
    # pre_model.load_weights(model_weights_path)



    #Model 2
    inp01 = Input(shape=(150, 150, 1))
    conv01 = Convolution2D(64, (11, 11), activation='relu', strides=4)(inp01)
    batchnorm01 = BatchNormalization()(conv01)
    maxp01 = MaxPooling2D((2, 2))(batchnorm01)

    zeropad0 = ZeroPadding2D((2,2))(maxp01)
    conv02 = Convolution2D(256, (5, 5), activation='relu', strides=1)(zeropad0)
    batchnorm02 = BatchNormalization()(conv02)
    maxp02 = MaxPooling2D((2, 2))(batchnorm02)

    zeropad01 = ZeroPadding2D((1,1))(maxp02)
    conv03 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad01)

    zeropad02 = ZeroPadding2D((1,1))(conv03)
    conv04 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad02)

    zeropad03 = ZeroPadding2D((1,1))(conv04)
    conv05 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad03)
    maxp03 = MaxPooling2D((2, 2))(conv05)

    flt01 = Flatten()(maxp03)
    dn01 = Dense(1024, activation='relu')(flt01)
    drp01 = Dropout(0.5)(dn01)

    dn02 = Dense(1024, activation='relu')(drp01)
    drp02 = Dropout(0.5)(dn02)
    op01 = Dense(1, activation='sigmoid')(drp02)
    pre_model1 = Model(input=inp01, output=op01)
    # pre_model1.load_weights(masks_weights_path)


    #Model3
    inp001 = Input(shape=(150, 150, 1))
    conv001 = Convolution2D(64, (11, 11), activation='relu', strides=4)(inp001)
    batchnorm001 = BatchNormalization()(conv001)
    maxp001 = MaxPooling2D((2, 2))(batchnorm001)

    zeropad00 = ZeroPadding2D((2,2))(maxp001)
    conv002 = Convolution2D(256, (5, 5), activation='relu', strides=1)(zeropad00)
    batchnorm002 = BatchNormalization()(conv002)
    maxp002 = MaxPooling2D((2, 2))(batchnorm002)

    zeropad001 = ZeroPadding2D((1,1))(maxp002)
    conv003 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad001)

    zeropad002 = ZeroPadding2D((1,1))(conv003)
    conv004 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad002)

    zeropad003 = ZeroPadding2D((1,1))(conv004)
    conv005 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad003)
    maxp003 = MaxPooling2D((2, 2))(conv005)

    flt001 = Flatten()(maxp003)
    dn001 = Dense(1024, activation='relu')(flt001)
    drp001 = Dropout(0.5)(dn001)

    dn002 = Dense(1024, activation='relu')(drp001)
    drp002 = Dropout(0.5)(dn002)
    op001 = Dense(1, activation='sigmoid')(drp002)
    pre_model2 = Model(input=inp001, output=op001)
    # pre_model2.load_weights("mass_CC_weights.h5")


    #Model4
    inp0001 = Input(shape=(150, 150, 1))
    conv0001 = Convolution2D(64, (11, 11), activation='relu', strides=4)(inp0001)
    batchnorm0001 = BatchNormalization()(conv0001)
    maxp0001 = MaxPooling2D((2, 2))(batchnorm0001)

    zeropad000 = ZeroPadding2D((2,2))(maxp0001)
    conv0002 = Convolution2D(256, (5, 5), activation='relu', strides=1)(zeropad000)
    batchnorm0002 = BatchNormalization()(conv0002)
    maxp0002 = MaxPooling2D((2, 2))(batchnorm0002)

    zeropad0001 = ZeroPadding2D((1,1))(maxp0002)
    conv0003 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad0001)

    zeropad0002 = ZeroPadding2D((1,1))(conv0003)
    conv0004 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad0002)

    zeropad0003 = ZeroPadding2D((1,1))(conv0004)
    conv0005 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad0003)
    maxp0003 = MaxPooling2D((2, 2))(conv0005)

    flt0001 = Flatten()(maxp0003)
    dn0001 = Dense(1024, activation='relu')(flt0001)
    drp0001 = Dropout(0.5)(dn0001)

    dn0002 = Dense(1024, activation='relu')(drp0001)
    drp0002 = Dropout(0.5)(dn0002)
    op0001 = Dense(1, activation='sigmoid')(drp0002)
    pre_model3 = Model(input=inp0001, output=op0001)
    # pre_model3.load_weights("mc_CC_weights.h5")


    #Model 5
    inp00001 = Input(shape=(150, 150, 1))
    conv00001 = Convolution2D(64, (11, 11), activation='relu', strides=4)(inp00001)
    batchnorm00001 = BatchNormalization()(conv00001)
    maxp00001 = MaxPooling2D((2, 2))(batchnorm00001)

    zeropad0000 = ZeroPadding2D((2,2))(maxp00001)
    conv00002 = Convolution2D(256, (5, 5), activation='relu', strides=1)(zeropad0000)
    batchnorm00002 = BatchNormalization()(conv00002)
    maxp00002 = MaxPooling2D((2, 2))(batchnorm00002)

    zeropad00001 = ZeroPadding2D((1,1))(maxp00002)
    conv00003 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad00001)

    zeropad00002 = ZeroPadding2D((1,1))(conv00003)
    conv00004 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad00002)

    zeropad00003 = ZeroPadding2D((1,1))(conv00004)
    conv00005 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad00003)
    maxp00003 = MaxPooling2D((2, 2))(conv00005)

    flt00001 = Flatten()(maxp00003)
    dn00001 = Dense(1024, activation='relu')(flt00001)
    drp00001 = Dropout(0.5)(dn00001)

    dn00002 = Dense(1024, activation='relu')(drp00001)
    drp00002 = Dropout(0.5)(dn00002)
    op00001 = Dense(1, activation='sigmoid')(drp00002)
    pre_model4 = Model(input=inp00001, output=op00001)
    # pre_model4.load_weights("mass_MLO_weights.h5")



    #Model 6
    inp000001 = Input(shape=(150, 150, 1))
    conv000001 = Convolution2D(64, (11, 11), activation='relu', strides=4)(inp000001)
    batchnorm000001 = BatchNormalization()(conv000001)
    maxp000001 = MaxPooling2D((2, 2))(batchnorm000001)

    zeropad00000 = ZeroPadding2D((2,2))(maxp000001)
    conv000002 = Convolution2D(256, (5, 5), activation='relu', strides=1)(zeropad00000)
    batchnorm000002 = BatchNormalization()(conv000002)
    maxp000002 = MaxPooling2D((2, 2))(batchnorm000002)

    zeropad000001 = ZeroPadding2D((1,1))(maxp000002)
    conv000003 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad000001)

    zeropad000002 = ZeroPadding2D((1,1))(conv000003)
    conv000004 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad000002)

    zeropad000003 = ZeroPadding2D((1,1))(conv000004)
    conv000005 = Convolution2D(256, (3, 3), activation='relu', strides=1)(zeropad000003)
    maxp000003 = MaxPooling2D((2, 2))(conv000005)

    flt000001 = Flatten()(maxp000003)
    dn000001 = Dense(1024, activation='relu')(flt000001)
    drp000001 = Dropout(0.5)(dn000001)

    dn000002 = Dense(1024, activation='relu')(drp000001)
    drp000002 = Dropout(0.5)(dn000002)
    op000001 = Dense(1, activation='sigmoid')(drp000002)
    pre_model5 = Model(input=inp000001, output=op000001)
    # pre_model5.load_weights("mass_MLO_weights.h5")









    inputgenerator=generate_generator_multiple(generator1=train_datagen_CC,
                                               generator2 = train_datagen_MLO,
                                               generator1mass=train_datagen_CC_mass,
                                               generator2mass = train_datagen_MLO_mass,
                                               generator1mc=train_datagen_CC_mc,
                                               generator2mc = train_datagen_MLO_mc,
                                               batch_size=1,
                                               train_features_CC=train_features_CC[train_index],
                                               train_features_MLO=train_features_MLO[train_index],
                                               mass_features_CC=mass_features_CC[train_index],
                                               mass_features_MLO=mass_features_MLO[train_index],
                                               mc_features_CC=mc_features_CC[train_index],
                                               mc_features_MLO=mc_features_MLO[train_index],
                                               train_labels = train_labels[train_index],
                                               )

    testgenerator=generate_generator_multiple(generator1=test_datagen_CC,
                                               generator2 = test_datagen_MLO,
                                               generator1mass=test_datagen_CC_mass,
                                               generator2mass = test_datagen_MLO_mass,
                                               generator1mc=test_datagen_CC_mc,
                                               generator2mc = test_datagen_MLO_mc,
                                               batch_size=1,
                                               train_features_CC=train_features_CC[test_index],
                                               train_features_MLO=train_features_MLO[test_index],
                                               mass_features_CC=mass_features_CC[test_index],
                                               mass_features_MLO=mass_features_MLO[test_index],
                                               mc_features_CC=mc_features_CC[test_index],
                                               mc_features_MLO=mc_features_MLO[test_index],
                                               train_labels = train_labels[test_index],
                                               )

    # print(flt4.shape)



    mrgCC = Merge(mode='concat')([drp2,drp02,drp002])

    mrgMLO = Merge(mode='concat')([drp0002, drp00002, drp000002])


    mrg = Merge(mode='concat')([mrgCC, mrgMLO])


    op = Dense(1, activation='sigmoid')(mrg)

    model = Model(input=[inp1, inp01, inp001, inp0001, inp00001, inp000001], output=op)



    # model.load_weights("sixcnnsweights4.h5")
    model.compile(optimizers.SGD(lr=1e-4, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = model.fit_generator(inputgenerator,
                            # samples_per_epoch=len(train_features[train_index]),
                            epochs=50,
                            steps_per_epoch = len(train_features_CC[train_index])/1
                            # verbose=0
                            )

    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    from scipy import interp
    Y_test = train_labels[test_index]
    y_pred = model.predict_generator(testgenerator, verbose=0, steps=len(train_features_CC[test_index]))
    # y_pred = (model.predict([train_features_CC[test_index], mass_features_CC[test_index], mc_features_CC[test_index], train_features_MLO[test_index],  mass_features_MLO[test_index], mc_features_MLO[test_index]]))
    print(y_pred)
    y_pred_around = np.around(y_pred)
    # y_pred_around = y_pred >= 0.3
    # y_pred_around = y_pred_around.astype(int)
    # print(Y_test)
    # print("========================================================================")
    # print(y_pred)
    print("Classification report")
    print(classification_report(Y_test, y_pred_around))
    print("_______________________________________________________________")
    print("Accuracy score")
    print("_____________________________________________________________")

    print(accuracy_score(y_true=Y_test, y_pred=y_pred_around))
    print("_____________________________________________________________")
    print("Confusion matrix")

    cm = confusion_matrix(Y_test, y_pred_around)
    print(cm)
    print("_____________________________________________________________")

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred.ravel())

    print(thresholds)


    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # plt.plot(fpr, tpr, lw=1, alpha=0.3,
    #          label='ROC fold %d (AUC = %0.2f)' % (count-1, roc_auc))


    # auc_keras = auc(fpr_keras, tpr_keras)
    # print(fpr_keras)
    # print(tpr_keras)
    # print(auc_keras)

    # plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    #
    # # plt.plot(fpr_keras[2], tpr_keras[2], color='darkorange',
    # #      lw=2, label='ROC curve (area = %0.2f)' % auc_keras[2])
    #
    #
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc="lower right")
    # plt.show()


    scores = model.evaluate_generator(testgenerator, steps=len(train_features_CC[test_index])/1)
    print("keras_score")
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    # print(scores[2])
    # print(scores[3])
    # plt.plot(history.history['loss'])
    # # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
