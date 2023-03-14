import numpy as np

def load_data():
    with open("simple-food-reviews.txt", "r") as data:
        allLines = [line.replace("\n","") for line in data.readlines()]
    return allLines

class reviewItem:
    def __init__(self,reviewLine):
        first_space_idx = reviewLine.find(" ")
        self.classified = reviewLine[:first_space_idx] == "1"
        self.review_comment = reviewLine[first_space_idx + 1:]

data = load_data()
reviews = [reviewItem(rev) for rev in data]
np.random.shuffle(reviews)

# for review in reviews:
#     print(("Classified: {}, Comment: {}").format(review.classified, review.review_comment))

training_data = reviews[:12]
test_data = reviews[12:]

# print("Training data:")
# for review in training_data:
#     print(("Classified: {}, Comment: {}").format(review.classified, review.review_comment))

# print("Test data")
# for review in test_data:
#     print(("Classified: {}, Comment: {}").format(review.classified, review.review_comment))

# we want word -> {numGood, numBad}
words = dict()
numgood = 0
numbad = 0
for review in training_data:
    if review.classified:
        numgood += 1
    else:
        numbad += 1

    comment = review.review_comment.split()
    for word in comment:
        if word in words:
            if review.classified:
                words[word][0] += 1
            else:
                words[word][1] += 1
        else:
            if review.classified:
                words[word] = [1, 0]
            else:
                words[word] = [0, 1]

# for word in words.keys():
#     print(("Word: {}, Good: {}, Bad: {}").format(word, words[word][0], words[word][1]))

numReviews = len(training_data)
pBad = numbad / numReviews
pGood = numgood / numReviews

#print(("Bad reviews: {}, Good reviews: {}").format(pBad,pGood))
k = 1
numClass = 2
for word in words.keys():
    classNumber = words[word]
    if classNumber[0] == 0 or numgood == 0:
        classNumber[0] == (classNumber[0] + k)/ (numgood + numClass*k)
    else:
        classNumber[0] = classNumber[0] / numgood
    
    if classNumber[1] == 0 or numbad == 0:
        classNumber[1] = (classNumber[1] + k)/ (numgood + numClass*k)
    else:
        classNumber[1] = classNumber[1] / numbad
    
    # print(("Word: {}, Good: {}, Bad: {}").format(word, words[word][0], words[word][1]))

theconfusion = np.zeros((2,2))
for review in test_data:
    pgood = 1
    pbad = 1

    for word in words:
        if word in review.review_comment:
            pgood = pgood * words[word][0]
            pbad = pbad * words[word][1]
        else:
            pgood = pgood * (1 - words[word][0])
            pbad = pbad * (1 - words[word][1])

    for word in review.review_comment.split():
        if word not in words:
            pbad = pbad * (k / (k * numClass))
    
    pgivenbad = (pbad* pBad) / (pbad * pBad + pgood * pGood)
    pgivengood = 1 - pgivenbad

    reviewPrediction = True

    if pgivenbad > pgivengood:
        reviewPrediction = False
        if reviewPrediction != review.classified:
            theconfusion[1,0] += 1
        else:
            theconfusion[0,1] += 1
    else:
        if review.classified:
            theconfusion[0,0] += 1
        else:
            theconfusion[1,1] += 1
    
print("P(Positive | Review) = {}, P(Negative | Review) = {}".format(pGood, pBad))
accuracy = np.trace(theconfusion) / len(test_data)
print(*theconfusion, sep="\n")
print("Accuracy", round(accuracy, 4) * 100)    