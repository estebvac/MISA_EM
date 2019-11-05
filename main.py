from impl import measures
import pandas as pd
import numpy as np

dices_em = []
dices_kmeans = []
times = []
iterations = []

for i in range(1, 6):
    if i == 2:
        dice, dice_km, time, iterat = measures.test(2, iterations=150, initialization='kmeans', tolerance=0.1, display=True, modlaity=1)
    else:
        dice, dice_km, time, iterat = measures.test(i, iterations=150, initialization='kmeans',
                                                tolerance=0.1, display=True)

    dices_em.append(dice)
    dices_kmeans.append(dice_km)
    times.append(time)
    iterations.append(iterat)



results_df = pd.DataFrame.from_dict(dices_em)
init_df = pd.DataFrame.from_dict(dices_kmeans)
statistics = pd.DataFrame(np.array([times, iterations]).transpose())
results_df = pd.concat((results_df, init_df, statistics), axis=1)
results_df.to_csv('results.csv')
