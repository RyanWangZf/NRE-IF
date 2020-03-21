# -*- coding: utf-8 -*-
import os
import pandas as pd

import pdb

prefix = "./demo"

files = os.listdir(prefix)

files = [os.path.join(prefix,f) for f in files]

df = None

for file in files:
    try:
        print(file)
        df_ = pd.read_csv(file, index_col=0)
        os.remove(file)
    except:
        continue

    if df is None:
        df = df_

    else:
        df = pd.concat([df,df_])

df = df.sort_values(by="influence")
df = df.reset_index(drop=True)
df.to_csv(os.path.join(prefix,"demo.csv"))

