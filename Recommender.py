import pickle
import operator
from ModelTraining import train

try:
    print("Files present")
    with open('datasets/cleaned_df.pkl', 'rb') as pickle_file:
        model_data = pickle.load(pickle_file)
    with open('models/model.pkl', 'rb') as pickle_file:
        matrix = pickle.load(pickle_file)
except:
    print("Files not present")
    train()
    with open('datasets/cleaned_df.pkl', 'rb') as pickle_file:
        model_data = pickle.load(pickle_file)
    with open('models/model.pkl', 'rb') as pickle_file:
        matrix = pickle.load(pickle_file)


def recommend(food, limit):
    if food in model_data['ingredient'].to_list():
        food_index = model_data['ingredient'].to_list().index(food)
        food_list = matrix[food_index]
        food_recc = {}
        count = 0
        for i in range(len(food_list)):
            name = model_data['ingredient'].iloc[food_list[i][0]]
            score = 100 - food_list[i][1]*100
            food_recc[name] = score
            count += 1
            if count == int(limit) + 1:
                break
        food_recc = sorted(food_recc.items(),
                           key=operator.itemgetter(1),
                           reverse=True)
        res = {}
        for i in range(len(food_recc)):
            res[i] = food_recc[i]
        return res
    else:
        return "Ingredient not found"
