# %%
import _lib.preprocess as prep
import pandas as pd

df = prep.simple_hils_swls()

hils_cols = ['_hilstotalt1',
             '_hilstotalt2',
             'harmony_t1',
             'harmony_t2']

swls_cols = ['_swlstotalt1',
             '_swlstotalt2',
             'satisfaction_t1',
             'satisfaction_t2']

# %%
df[hils_cols].head()

# %%
df[swls_cols].head()
# %%
