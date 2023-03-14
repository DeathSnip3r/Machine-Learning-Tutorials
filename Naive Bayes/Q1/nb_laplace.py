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