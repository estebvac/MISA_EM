from impl import measures
import pandas as pd

dices_em = []
dices_kmeans = []

for i in range(1,6):
    dice, dice_km = measures.test(i,iterations = 10)
    dices_em.append(dice)
    dices_kmeans.append(dice_km)

dice_em_dataframe = pd.DataFrame.from_dict(dices_em)
dice_km_dataframe = pd.DataFrame.from_dict(dices_kmeans)
dices_dataframe = pd.concat([dice_em_dataframe, dice_km_dataframe], axis=1, sort=False)
dices_dataframe.to_csv('results.csv')
