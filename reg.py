import statsmodels.api as sm
import numpy as np
import pandas as pd

df = pd.read_csv("./prices.txt", sep="\\s+", header=None, index_col=None)
df.index = np.arange(df.shape[0])
df.rename(columns=lambda c: str(c), inplace=True)
print(df)

pct = df.pct_change().dropna()

offset = 300
Xs = [
    pct[str(i)].shift(k).iloc[offset:] for i in range(0, 50) for k in range(offset)
]

print(len(Xs))
print(Xs[:2])

for i in range(len(Xs)):
    i_instr = i // offset
    for j in range(i+1, len(Xs)):
        j_instr = j // offset
        for instr in range(50):
            if instr == i_instr or instr == j_instr:
                continue

            X = np.dstack([Xs[i], Xs[j]])[0]
            X = sm.add_constant(X)
            model = sm.OLS(pct[str(instr)].iloc[offset:], X)
            res = model.fit()
            # print("Done ", i, j)
            if res.rsquared > 0.1 and res.rsquared != 1:
                print(instr, i, j, res.rsquared)