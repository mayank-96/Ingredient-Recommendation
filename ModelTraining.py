# Import Libraries
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy import spatial
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import math
import pickle
import warnings
warnings.filterwarnings("ignore")


class Model:

    def __init__(self, data_path) -> None:
        # Read CSV File
        print("Reading data file ...")
        self.data = pd.read_csv(data_path)

    def model_data(self) -> None:
        print("Cleaning data....")
        # Cleaning data
        self.data['region'].iloc[110] = 'unknown_region'  # Handle nan value
        for i in range(255):
            if self.data['flavor_profile'].iloc[i] == '-1':
                self.data['flavor_profile'].iloc[i] = 'unknown_flavor'
            if self.data['state'].iloc[i] == '-1':
                self.data['state'].iloc[i] = 'unknown_state'
            if self.data['region'].iloc[i] == '-1':
                self.data['region'].iloc[i] = 'unknown_region'
            self.data['name'].iloc[i] = "_".join(
                self.data['name'].iloc[i].split()).lower()
            self.data['diet'].iloc[i] = "_".join(
                self.data['diet'].iloc[i].split()).lower()
            self.data['course'].iloc[i] = "_".join(
                self.data['course'].iloc[i].split()).lower()
            self.data['state'].iloc[i] = "_".join(
                self.data['state'].iloc[i].split()).lower()
            self.data['region'].iloc[i] = "_".join(
                self.data['region'].iloc[i].split()).lower()

        # One-Hot Encoding
        temp = self.data[['name', 'diet',
                          'flavor_profile', 'course', 'state', 'region']]
        one_hot_df = pd.get_dummies(temp).reset_index()

        # Finding all Ingredients
        ing = []
        for i in range(255):
            self.data['ingredients'].iloc[i] = self.data['ingredients'].iloc[i].split(
                ',')
            ing.append(self.data['ingredients'].iloc[i])

        # Reset data index
        self.data = self.data.reset_index()

        # Create ingredients df
        values = {'index': [], 'ingredient': []}
        for i in range(255):
            for j in ing[i]:
                values['index'].append(i)
                values['ingredient'].append("_".join(j.split()).lower())

        ing_df = pd.DataFrame(values)

        # Merge ingredients df and one-hot df
        final_data = pd.merge(ing_df, one_hot_df, how='inner', on='index')
        final_data.pop('index')

        # Clean data
        change = {
            'almonds': 'almond',
            'badam': 'almond',
            'aloo': 'potato',
            'potatoes': 'potato',
            'arbi_ke_patte': 'colocasia_leave',
            'arhar_dal': 'pigeon_pea',
            'atta': 'flour',
            'baingan': 'brinjal',
            'bell_peppers': 'bell_pepper',
            'besan_flour': 'besan',
            'bhuna_chana': 'roasted_chickbeans',
            'carrots': 'carrot',
            'cashew_nuts': 'cashews',
            'chana_dal': 'chana_daal',
            'chhena': 'chenna',
            'chilli': 'chillies',
            'chole': 'chickpeas',
            'drumsticks': 'drumstick',
            'fish_fillets': 'fish_fillet',
            'gobi': 'cauliflower',
            'green_chilies': 'green_chili',
            'green_chilli': 'green_chili',
            'green_chillies': 'green_chili',
            'green': 'greens',
            'gur': 'jaggery',
            'imli': 'tamarind',
            'kasuri_methi': 'fenugreek',
            'litre_milk': 'milk',
            'mustard_seed': 'mustard_seeds',
            'peanut': 'peanuts',
            'potol': 'pointed_gourd',
            'red_chilli': 'red_chili',
            'red_chillies': 'red_chili',
            'shimla_mirch': 'capsicum',
            'tomatoes': 'tomato',
            'yogurt': 'yoghurt'
        }
        for i in change:
            final_data['ingredient'] = final_data['ingredient'].replace([
                                                                        i], change[i])

        # Create model df
        model_df = final_data.groupby(['ingredient']).sum().reset_index()
        col = model_df.columns.to_list()
        col.remove('ingredient')
        ing_df = pd.DataFrame(model_df['ingredient'].to_list())

        # Convert all non-zero elements to one and merge it with ing df
        df = pd.merge(ing_df, model_df[col].astype(bool).astype(
            int), left_index=True, right_index=True)
        df = df.rename(columns={0: 'ingredients'})

        self.data = df

    def model(self) -> None:
        print("Training model....")
        variables = self.data.columns.to_list()
        variables.remove('ingredients')

        # Similarity Matrix
        matrix = []
        for i in range(self.data.shape[0]):
            scores = {}
            for j in range(self.data.shape[0]):
                # Calculating cosine distance
                score = spatial.distance.cosine(
                    self.data[variables].iloc[i], self.data[variables].iloc[j])
                if math.isnan(score):
                    scores[j] = -1
                else:
                    scores[j] = score
            scores = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
            matrix.append(scores)

        # Save final dataset
        # Open a file and use dump()
        with open('datasets/cleaned_df.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(self.data, file)

        # Save Matrix
        # Open a file and use dump()
        with open('models/model.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(matrix, file)

# Main Function


def train():
    ob = Model('datasets\indian_food.csv')

    # Pipeline
    ob.model_data()
    ob.model()
