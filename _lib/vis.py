import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from _lib import preprocess as pp
from wordcloud import WordCloud


def wordcloud(construct, interval, max_words=20):
    """Generates a wordcloud from the text of a interval
    on a scale

    Args:
        construct (str): one of ['hils', 'swls']
        interval (list): a list of two integers [start, end]
        max_words (int): amount of words in wordcloud. Defaults to 20.
    """
    df = pp.time_invariant_oc(construct=construct)
    df = df[(df[construct] >= interval[0]) & (df[construct] <= interval[1])]
    string, _ = pp.string_set(df)
    wordcloud = WordCloud(
        max_words=max_words,
        margin=10
        ).generate(string)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def deltas(show=False):
    """Creates a bar graph with the deltas for hils and swls

    Args:
        show (bool): Draw the graph or not. Defaults to False.

    Returns:
        figure: A figure with the bar graph
    """
    data_frame, _ = pp.hils_swls_metrics()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data_frame['delta_hils'],
                               name='hils'))
    fig.add_trace(go.Histogram(x=data_frame['delta_swls'],
                               name='swls'))
    fig.update_layout(bargap=.2,
                      bargroupgap=.1,
                      title_text='Deltas for hilsand swls',
                      xaxis_title_text='Value',
                      yaxis_title_text='Count')
    fig.update_traces(opacity=.50)
    if show:
        fig.show()
    return fig


def pre_post(construct, show=False):
    """Creates a pre post scatterplot and adds all the metrics to it.

    Args:
        construct (str): ['hils', 'swls']
        show (bool): Draw the figure or not. Defaults to False.

    Returns:
        figure: A figure with the scatterplot.
    """
    data_frame, stats = pp.hils_swls_metrics()
    words = 'harmony' if construct == 'hils' else 'satisfaction'
    d, p, r = (np.round(stats[f'd_{construct}'], 4),
               np.round(stats[f'p_{construct}'], 4),
               np.round(stats[f'r_{construct}'], 4))
    title = f'{construct}, pre vs post<br>d = {d}<br>p = {p}<br>r = {r}'
    line_dict = dict(
        color='black',
        width=1,
        dash='dash'
    )
    fig = px.scatter(data_frame=data_frame,
                     x=f'_{construct}totalt1',
                     y=f'_{construct}totalt2',
                     color=f'rel_{construct}',
                     marginal_x='histogram',
                     marginal_y='histogram',
                     # color_discrete_sequence=px.colors.sequential.Viridis,
                     # hover_name='Metrics',
                     hover_data=[f'RCI_{construct}',
                                 f'SID_{construct}',
                                 f'delta_{construct}',
                                 f'{words}_t1',
                                 f'{words}_t2'])
    fig.add_traces([
        go.Scatter(
            x=data_frame[f'_{construct}totalt1'],
            y=data_frame[f'_{construct}totalt1'] - stats[f'CI_{construct}'],
            showlegend=False,
            mode='lines',
            line=line_dict
        ),
        go.Scatter(
            x=data_frame[f'_{construct}totalt1'],
            y=data_frame[f'_{construct}totalt1'] + stats[f'CI_{construct}'],
            showlegend=False,
            mode='lines',
            line=line_dict
        )]
    )
    fig.update_layout(title_text=title,
                      xaxis_title_text='Pre',
                      yaxis_title_text='post')
    if show:
        fig.show()
    return fig


def pred_label(pred, label):
    fig = px.scatter(
        x=label,
        y=pred
    )
    fig.show()
