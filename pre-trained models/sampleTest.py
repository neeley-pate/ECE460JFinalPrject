import preprocessing

X, Y = preprocessing.sampleImages(10, 10, 10)
X1, Y1 = preprocessing.sampleImages(10, 10, 10)

print(X == X1)
print(Y == Y1)