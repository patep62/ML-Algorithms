import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import math
np.random.seed(1342)


filePath1 = "C:\\Users\\ppart\\OneDrive\\Desktop\\School Stuff\\4SL4\\Assignment5\\moraine-lake.jpg"
filePath2 = "C:\\Users\\ppart\\OneDrive\\Desktop\\School Stuff\\4SL4\\Assignment5\\fox.jpg"

image = mpimg.imread(filePath2)
#plt.imshow(image)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))
 
# Convert to float type
pixel_vals = np.float32(pixel_vals)

#Function computing euclidian distance
def getDistance(x1,x2,y1,y2,z1,z2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

#Function that given a set of clusters, will calculate the mid point in each cluster.
def getNewCentres(clusters, k):

    newCentres = np.zeros((k, 3))
    for i in range(k):
        if clusters[i]:
            newCentres[i] = np.mean(clusters[i], axis=0)
    return newCentres

#Kmeans training algorithm
def kmeans(X, k, max_iter=100, initialization="random"):
    
    #Shuffle pixels to randomly select a set of centres.
    shuffledPixels = np.random.permutation(X)

    #Initialization method to ensure a minimum distance between random centres.
    if initialization == "randomWithDistance":
        minimumDistanceThreshold = 30 #Minimum distance
        centres = [shuffledPixels[0]]

        #For each pixel, check its distance against each existing centre. If large enough, append to centres list.
        for point in shuffledPixels:
            if (len(centres) == k):
                break
            tooClose = False
            for centre in centres:
                dist = getDistance(point[0],centre[0],point[1],centre[1],point[2],centre[2])
                if dist <= minimumDistanceThreshold:
                    tooClose = True
                    break
            if not tooClose:
                centres.append(point)

    else:
        centres =  shuffledPixels[:k] #Default initialization, just take the first k shuffles pixels.

    #Initialize a cluster list to hold a set of points assigned to each cluster.
    clusters = [[] for i in range(k)]
    newImg = np.zeros(shape=(X.shape)) #The new image that will be returned.
    
    #Triple nested for loop, running for m iterations, through each pixel, then through each centre.
    for m in range(max_iter):
        mse = MSE(X, newImg)
        print("Iteration: " + str(m) + " MSE: " + str(mse))
        for i in range(X.shape[0]):
            lowestDist = 1000000000
            closestCentre = 0
            for j in range(k):
                #Get the distance from the pixel to each centre to find the closest centre.
                dist = getDistance(X[i][0],centres[j][0],X[i][1],centres[j][1],X[i][2],centres[j][2])
                if dist < lowestDist:
                    closestCentre = j
                    lowestDist = dist
            #Add the pixel to its cluster, and also overwrite the original pixel with the new centre it was assigned to.
            clusters[closestCentre].append(X[i])
            newImg[i] = centres[closestCentre]
        
        #Get our new centres.
        newCentres = getNewCentres(clusters, k)
        #If they equal our old centres, exit the alogrithm.
        if np.array_equal(np.round(newCentres), np.round(centres)): #Need to round or else will take very long to equal exactly.
            print("breaked at: " + str(m+1))
            break
        centres = newCentres

    return newImg

#Get the MSE between original and reproduced image.
def MSE(a, b):
    return np.mean((a - b)**2)

reconstructedImg = kmeans(pixel_vals, 3)
print(str(MSE(pixel_vals, reconstructedImg)))

#Convert back to 8 bit format and reshape to orinal image size.
reconstructedImg = np.uint8(reconstructedImg)
plt.imshow(reconstructedImg.reshape(image.shape))
plt.show()
