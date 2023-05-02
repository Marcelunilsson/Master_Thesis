# %%
import _lib.preprocess as pp
import _lib.vis as vis


df = pp.simple_hils_swls()

df, stats = pp.add_metrics(df=df)
df['delta_hils'] = pp.add_deltas(df,
                                 'hils')
df['delta_swls'] = pp.add_deltas(df,
                                 'swls')

long_df = pp.long_df(df)

fig1 = vis.pre_post(
    data_frame=df,
    construct='hils',
    stats=stats
    )
fig1.write_html(
    '_graphs/pre_post_hils.html'
)
fig2 = vis.pre_post(
    data_frame=df,
    construct='swls',
    stats=stats
    )
fig2.write_html(
    '_graphs/pre_post_swls.html'
)
fig3 = vis.deltas(
    data_frame=df
    )
fig3.write_html(
    '_graphs/deltas_hils_swls.html'
)
fig4 = vis.facet(
    long_df
    )
fig4.write_html(
    '_graphs/facet_hils_swls.html'
)

with open('_graphs/all_graphs.html', 'w') as f:
    f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))
# %%
