import numpy as np

# Function to open file, read each line from file into an array allLines 
def load_data():
    with open("simple-food-reviews.txt", "r") as data:
        allLines = [line.replace("\n","") for line in data.readlines()]
    return allLines

# Object which will be used to simplify implementation
class reviewItem:
    def __init__(self,reviewLine):
        first_space_idx = reviewLine.find(" ")
        self.classified = reviewLine[:first_space_idx] == "1"
        self.review_comment = reviewLine[first_space_idx + 1:]

# Loading data and creating an array of reviewItem objets from data array
data = load_data()
reviews = [reviewItem(rev) for rev in data]
#np.random.shuffle(reviews)

training_data = reviews[:12]
test_data = reviews[12:]
# print("Training data:")
# for review in training_data:
#     print(("Classified: {}, Comment: {}").format(review.classified, review.review_comment))

# Generate frequency 'table' using a dict, key is the word and the data will be a tuple: {#goodReview,#badReview}
words = dict()
countGood = 0
countBad = 0
for item in training_data:
    if item.classified:
        countGood += 1
    else:
        countBad += 1
    for word in item.review_comment.split():
        # if the word is in the dictionary, just increase the count in relevant place
        if word in words:
            if item.classified:
                words[word][0] += 1
            else:
                words[word][1] += 1
        else:
            if item.classified:
                words[word] = [1,0]
            else:
                words[word] = [0,1]

# for word in words:
#     print(("{} , {}").format(word,words[word]))
# print("NumGood: {}, NumBad: {}".format(countGood, countBad))

# Variables for later
numberReviews = len(training_data)
pGoodReview = countGood / numberReviews # Priors
pBadReview = countBad / numberReviews # Priors
k = 1
nClass = 2

# calculate probabilities in the table, if there is a 0 probability, use laplace smoothing
for word in words.keys():
    wordCount = words[word]
    if wordCount[0] == 0 or countGood == 0:
        wordCount[0] = (wordCount[0] + k) / (countGood + k*nClass) #laplace smoothing
    else:
        wordCount[0] = wordCount[0] / countGood

    if wordCount[1] == 0 or countBad == 0:
        wordCount[1] = (wordCount[1] + k) / (countBad + k*nClass)
    else:
        wordCount[1] = wordCount[1] / countBad
    	
    if wordCount[0] == wordCount[1]:
        wordCount[0] = (wordCount[0] + k) / (countGood + k*nClass)
        wordCount[1] = (wordCount[1] + k) / (countBad + k*nClass)

# for word in words:
#     print(("{} , {}").format(word,words[word]))


# We have now trained our model. Can now test our model
theConfusion = np.zeros((2,2))
for item in test_data:
    pGoodNew = 1
    pBadNew = 1
    # Generate encoding
    arr = []
    for word in words.keys():
        if word in item.review_comment.split():
            arr.append(True)
        else:
            arr.append(False)

    # For all words that are in the dictionary, compute P(Word | Good) and P(Word | Bad)
    index = 0
    for word in words.keys():
        if arr[index] == True:
            pGoodNew = pGoodNew * words[word][0]
            pBadNew = pBadNew * words[word][1]
            index += 1
        else:
            pGoodNew = pGoodNew * (1 - words[word][0])
            pBadNew = pBadNew * (1 - words[word][1]) 
            index += 1

    # # Account for unseen words: P(UnseenWord | Good) and P(UnseenWord | Bad)
    for word in item.review_comment.split():
        if word not in words.keys():
            pGoodNew = pGoodNew * (k / (k*nClass))
            pBadNew = pBadNew * (k / (k*nClass))

    pGivenBad = np.abs((pBadNew * pBadReview) / (pBadNew * pBadReview + pGoodNew * pGoodReview))
    pGivenGood = 1 - pGivenBad
    print(("P(Bad | Review) = {}, P(Good | Review) = {}").format(pGivenBad,pGivenGood))

    # After classification, we want to see the results
    if pGivenBad > pGivenGood:
        review = False
        if review == item.classified:
            theConfusion[0,1] += 1
        else:
            theConfusion[1,0] += 1
    else:
        if item.classified:
            theConfusion[0,0] += 1
        else:
            theConfusion[1,1] += 1

# Print results of testing
accuracy = np.trace(theConfusion) / len(test_data)
print(*theConfusion, sep="\n")
print("Accuracy: ", round(accuracy,4) * 100)