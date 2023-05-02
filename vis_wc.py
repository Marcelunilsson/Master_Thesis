# %%
import _lib.lime_survey as ls
import _lib.preprocess as pp
import pandas as pd
import plotly.express as px
# %%
df, stats = pp.hils_swls_metrics()

ls.questionaire(200, 'hils', color='black', collage_color=False)

# %%

constructs = ['hils', 'swls']
construct = constructs[1]
df = pp.time_invariant_oc(construct=construct)
text, set = pp.string_set(df, 'words')

# %%
word_df = pd.DataFrame()

for i, row in df.iterrows():
    word_list = row['words'].split()
    word_dict = {
        'word': word_list,
        construct: [row[construct]]*len(word_list)
    }
    word_df = pd.concat([word_df, pd.DataFrame(word_dict)],
                        ignore_index=True)
# %%
fig = px.histogram(
    data_frame=word_df,
    x=construct,
    color='word',
    barmode='overlay',
    opacity=0.5,
    marginal='violin'
)
fig.write_html('_graphs/swls_bar.html')
# %%
fig.show()

# %%

df = pd.read_csv('_data/Big_swls_delta_set.csv')
# %%
