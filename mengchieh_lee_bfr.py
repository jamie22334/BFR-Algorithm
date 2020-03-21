import csv
import sys
import time
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

start = time.time()

filePath = sys.argv[1]

givenClusterCount = int(sys.argv[2])
# print("given cluster: " + str(givenClusterCount))

dataList = list()
# data point -> data_id
dataMap = dict()
solutionCluster = list()

with open(filePath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    # (data_id, cluster_id, x0...x9)
    for row in csv_reader:
        dataList.append(row[2:])
        dataTuple = tuple(row[2:])
        dataMap[dataTuple] = row[0]
        solutionCluster.append(row[1])


totalData = len(dataList)
# print("total data: " + str(totalData))
percentage = 0.2

modData = totalData % round(totalData * percentage)
initData = dataList[0: round(totalData * percentage) + modData]
# print("init data count: " + str(len(initData)))
labels = KMeans(n_clusters=givenClusterCount * 10, random_state=0).fit_predict(initData)

# cluster_label -> data point
labelMap = dict()
for i in range(len(initData)):
    label = labels[i]
    if label not in labelMap:
        labelMap[label] = list()
    dataTuple = tuple(initData[i])
    labelMap[label].append(dataTuple)

# remove cluster with 10 points
remainingData = list()
for key in labelMap:
    if len(labelMap[key]) <= 10:
        for dataTuple in labelMap[key]:
            dataPoint = list(dataTuple)
            initData.remove(dataPoint)
            remainingData.append(dataPoint)

# print("new data count: " + str(len(initData)))
dsCount = len(initData)

# k-means with regular k
labels = KMeans(n_clusters=givenClusterCount, random_state=0).fit_predict(initData)

# generate DS
clusterMap = dict()
for i in range(len(initData)):
    label = labels[i]
    if label not in clusterMap:
        num = 1
        sumV = np.array(initData[i], dtype=float)
        sumQ = np.square(sumV)
        idList = list()
        idList.append(dataMap[tuple(initData[i])])

        clusterMap[label] = (num, sumV, sumQ, idList)
    else:
        num = clusterMap[label][0]
        sumV = clusterMap[label][1]
        sumQ = clusterMap[label][2]
        idList = clusterMap[label][3].copy()

        num += 1
        tmpSum = np.array(initData[i], dtype=float)
        sumV = np.add(sumV, tmpSum)
        sumQ = np.add(sumQ, np.square(tmpSum))

        idList.append(dataMap[tuple(initData[i])])
        clusterMap[label] = (num, sumV, sumQ, idList)

# generate CS + RS
# print("remaining data count: " + str(len(remainingData)))
labels = KMeans(n_clusters=round(len(remainingData) / 2), random_state=0).fit_predict(remainingData)
labelMap.clear()
for i in range(len(remainingData)):
    label = labels[i]
    dataTuple = tuple(remainingData[i])
    if label not in labelMap:
        labelMap[label] = list()
    labelMap[label].append(dataTuple)

rsList = list()
csList = list()
csCount = 0
for key in labelMap:
    # RS
    if len(labelMap[key]) <= 2:
        for dataTuple in labelMap[key]:
            dataPoint = list(dataTuple)
            rsList.append(dataPoint)

    # generate CS
    else:
        num = len(labelMap[key])
        sumV = np.zeros(10)
        sumQ = np.zeros(10)
        idList = list()
        for dataTuple in labelMap[key]:
            tmpSum = np.array(list(dataTuple), dtype=float)
            sumV = np.add(sumV, tmpSum)
            sumQ = np.add(sumQ, np.square(tmpSum))
            idList.append(dataMap[dataTuple])
        csList.append((num, sumV, sumQ, idList))
        csCount += num

intermediateResult = list()
intermediateResult.append((dsCount, len(csList), csCount, len(rsList)))

print("round1 DS: " + str(dsCount))
print("round1 CS length: " + str(len(csList)))
print("round1 CS: " + str(csCount))
print("round1 RS: " + str(len(rsList)))

startIndex = round(totalData * percentage) + modData
endIndex = startIndex + round(totalData * percentage)
roundNum = 2

while endIndex <= totalData:
    # print("start: " + str(startIndex) + " end: " + str(endIndex))
    initData = dataList[startIndex: endIndex]
    # print("init data count: " + str(len(initData)))
    for dataPoint in initData:
        point = np.array(dataPoint, dtype=float)
        nearCluster = ""
        nearDistance = 100
        # print("data point: " + str(point))
        for c in clusterMap:
            num = clusterMap[c][0]
            sumV = clusterMap[c][1]
            sumQ = clusterMap[c][2]
            centroid = sumV / num
            variance = sumQ / num - np.square(centroid)
            deviation = np.sqrt(variance)
            normalized = np.square((point - centroid) / deviation)
            maDistance = math.sqrt(normalized.sum(axis=0))
            if maDistance < 2 * math.sqrt(10):
                if maDistance < nearDistance:
                    nearDistance = maDistance
                    nearCluster = c
        if nearDistance < 100:
            # assign to DS
            num = clusterMap[nearCluster][0]
            sumV = clusterMap[nearCluster][1]
            sumQ = clusterMap[nearCluster][2]
            idList = clusterMap[nearCluster][3].copy()

            num += 1
            tmpSum = np.array(dataPoint, dtype=float)
            sumV = np.add(sumV, tmpSum)
            sumQ = np.add(sumQ, np.square(tmpSum))

            idList.append(dataMap[tuple(dataPoint)])
            clusterMap[nearCluster] = (num, sumV, sumQ, idList)
            dsCount += 1

        else:
            nearIndex = -1
            nearDistance = 100

            for index in range(len(csList)):
                num = csList[index][0]
                sumV = csList[index][1]
                sumQ = csList[index][2]
                centroid = sumV / num
                variance = sumQ / num - np.square(centroid)
                deviation = np.sqrt(variance)
                np.seterr(divide='ignore')
                normalized = np.square((point - centroid) / deviation)
                maDistance = math.sqrt(normalized.sum(axis=0))
                if maDistance < 2 * math.sqrt(10):
                    if maDistance < nearDistance:
                        nearDistance = maDistance
                        nearIndex = index

            if nearDistance < 100:
                # assign to CS
                num = csList[nearIndex][0]
                sumV = csList[nearIndex][1]
                sumQ = csList[nearIndex][2]
                idList = csList[nearIndex][3].copy()

                num += 1
                tmpSum = np.array(dataPoint, dtype=float)
                sumV = np.add(sumV, tmpSum)
                sumQ = np.add(sumQ, np.square(tmpSum))

                idList.append(dataMap[tuple(dataPoint)])
                csList[nearIndex] = (num, sumV, sumQ, idList)
                csCount += 1
            else:
                # assign to RS
                rsList.append(dataPoint)

    # k-means on RS to generate CS
    # print("temp RS count: " + str(len(rsList)))
    if len(rsList) >= 4:
        labels = KMeans(n_clusters=round(len(rsList) / 2), random_state=0).fit_predict(rsList)
        labelMap.clear()
        for i in range(len(rsList)):
            label = labels[i]
            dataTuple = tuple(rsList[i])
            if label not in labelMap:
                labelMap[label] = list()
            labelMap[label].append(dataTuple)

        for key in labelMap:
            # RS
            if len(labelMap[key]) <= 2:
                dataPoint = list(labelMap[key][0])
            # generate CS
            else:
                num = len(labelMap[key])
                sumV = np.zeros(10)
                sumQ = np.zeros(10)
                idList = list()
                for dataTuple in labelMap[key]:
                    tmpSum = np.array(list(dataTuple), dtype=float)
                    sumV += tmpSum
                    sumQ += np.square(tmpSum)
                    idList.append(dataMap[dataTuple])
                    rsList.remove(list(dataTuple))
                csList.append((num, sumV, sumQ, idList))
                csCount += num

    # merge CS
    # print("before merge length: " + str(len(csList)))
    mergedPair = list()
    needMerge = False
    for i in range(len(csList)):
        xCentroid = csList[i][1] / csList[i][0]
        for j in range(i + 1, len(csList)):
            num = csList[j][0]
            sumV = csList[j][1]
            sumQ = csList[j][2]
            centroid = sumV / num
            variance = sumQ / num - np.square(centroid)
            deviation = np.sqrt(variance)
            np.seterr(divide='ignore')
            normalized = np.square((xCentroid - centroid) / deviation)
            maDistance = math.sqrt(normalized.sum(axis=0))

            if maDistance < 2 * math.sqrt(10):
                mergedPair.append((i, j))
                needMerge = True
                break
        if needMerge:
            break

    # how to update csList
    while len(mergedPair) > 0:
        pair = mergedPair[0]
        i = pair[0]
        j = pair[1]

        num = csList[i][0] + csList[j][0]
        sumV = np.add(csList[i][1], csList[j][1])
        sumQ = np.add(csList[i][2], csList[j][2])
        idList = csList[i][3].copy()
        idList.extend(csList[j][3])

        csList[i] = (num, sumV, sumQ, idList)
        del csList[j]
        mergedPair.remove(pair)

        needMerge = False
        for i in range(len(csList)):
            xCentroid = csList[i][1] / csList[i][0]
            for j in range(i + 1, len(csList)):
                num = csList[j][0]
                sumV = csList[j][1]
                sumQ = csList[j][2]
                centroid = sumV / num
                variance = sumQ / num - np.square(centroid)
                deviation = np.sqrt(variance)
                np.seterr(divide='ignore')
                normalized = np.square((xCentroid - centroid) / deviation)
                maDistance = math.sqrt(normalized.sum(axis=0))

                if maDistance < 2 * math.sqrt(10):
                    mergedPair.append((i, j))
                    needMerge = True
                    break
            if needMerge:
                break

    startIndex = endIndex
    endIndex = startIndex + round(totalData * percentage)

    # last round
    # merge CS with DS
    if endIndex > totalData:
        copyCsList = csList.copy()
        csList.clear()

        for cs in copyCsList:
            doMerge = False
            num = cs[0]
            sumV = cs[1]
            sumQ = cs[2]
            centroid = sumV / num
            variance = sumQ / num - np.square(centroid)
            deviation = np.sqrt(variance)

            for ds in clusterMap:
                # sum / n
                dsCentroid = clusterMap[ds][1] / clusterMap[ds][0]
                np.seterr(divide='ignore')
                normalized = np.square((dsCentroid - centroid) / deviation)
                maDistance = math.sqrt(normalized.sum(axis=0))

                if maDistance < 2 * math.sqrt(10):
                    num += clusterMap[ds][0]
                    sumV = np.add(sumV, clusterMap[ds][1])
                    sumQ = np.add(sumQ, clusterMap[ds][2])
                    idList = cs[3].copy()
                    idList.extend(clusterMap[ds][3])

                    clusterMap[ds] = (num, sumV, sumQ, idList)

                    doMerge = True
                    # preCsCount = csCount
                    csCount -= cs[0]
                    dsCount += cs[0]
                    # print(str(preCsCount) + " - " + str(cs[0]) + " = " + str(csCount))
                    break
            if not doMerge:
                csList.append(cs)

    # last round
    if endIndex > totalData:
        intermediateResult.append((dsCount, 0, 0, len(rsList) + csCount))
    else:
        intermediateResult.append((dsCount, len(csList), csCount, len(rsList)))
    # print("round" + str(roundNum) + " DS: " + str(dsCount))
    # print("round" + str(roundNum) + " CS length: " + str(len(csList)))
    # print("round" + str(roundNum) + " CS: " + str(csCount))
    # print("round" + str(roundNum) + " RS: " + str(len(rsList)))
    # print("round" + str(roundNum) + " total: " + str(dsCount + csCount + len(rsList)))
    roundNum += 1


outputList = list()
for key in clusterMap:
    idList = clusterMap[key][3]
    for dataId in idList:
        outputList.append((int(dataId), int(key)))

for cs in csList:
    idList = cs[3]
    for dataId in idList:
        outputList.append((int(dataId), -1))

for rs in rsList:
    dataTuple = tuple(rs)
    dataId = dataMap[dataTuple]
    outputList.append((int(dataId), -1))

outputList.sort()

myCluster = list()
for x in outputList:
    myCluster.append(x[1])

score = normalized_mutual_info_score(myCluster, solutionCluster)
print("score: " + str(score))


outputFileName = sys.argv[3]
with open(outputFileName, "w") as fp:
    fp.write("The intermediate results:\n")
    for i in range(len(intermediateResult)):
        fp.write("Round " + str(i + 1) + ": ")
        fp.write(','.join(str(y) for y in intermediateResult[i]))
        fp.write("\n")

    fp.write("\nThe clustering results:\n")
    for x in outputList:
        fp.write(','.join(str(y) for y in x))
        fp.write("\n")

end = time.time()
print('Duration: ' + str(end - start))
