import pandas as pd
import numpy as np

from scipy.stats import binom

import streamlit as st
import plotly.express as px


grid_points = st.sidebar.slider(
    label='# of grid points (each point = p)',
    min_value=0,
    max_value=100,
    value=100
    )

k = st.sidebar.slider(
    label='Water samples (w)',
    min_value=0,
    max_value=10,
    value=6
    )

n = st.sidebar.slider(
    label='Total (Water + Land) (n)',
    min_value=0,
    max_value=10,
    value=9
    )

p = st.sidebar.slider(
    label='Probability (p)',
    min_value=0.0,
    max_value=1.0,
    value=0.7
    )

# @st.cache(persist=True, show_spinner=True)
def pr_p_given_data(k, n, grid_point):
    
    p_grid = np.linspace(0, 1, grid_point)

    binom_distribution = binom.pmf(k, n=n, p=p_grid)

    df = pd.DataFrame(data={'prob_p': p_grid, 'prob_w': binom_distribution})
    
    return p_grid, binom_distribution, df

def pr_data_given_p(n, p):
    k = np.arange(0, n+1, 1)

    binom_distribution = binom.pmf(k, n, p)

    df = pd.DataFrame(data={'k': k, 'prob': binom_distribution})
    
    return df

st.text('Page 32 of the book "Statistical Rethinking 2ed" by Richard McElreath')
st.markdown("""In this case, once we add our assumptions that (1) every toss is independent of the other
tosses and (2) the probability of W is the same on every toss, probability theory provides
a unique answer, known as the binomial distribution. This is the common “coin tossing”
distribution. And so the probability of observing W waters and L lands, with a probability p
of water on each toss, is:""")
st.latex(r'''
Pr(W, L \mid p) =  \frac{(W+L)!}{W!L!} p^W (1 − p)^{L}
''')
st.markdown('''
Read the above as:

The counts of “water” W and “land’ L are distributed binomially, with probability p of “water” on each toss.
''')

p_grid, binom_distribution, df = pr_p_given_data(k, n, grid_points)

fig_pl = px.area(df, x="prob_p", y="prob_w")
fig_pl.update_yaxes(range=[0, 1])
fig_pl.add_annotation(
    text=f"""Water samples: {k}
    Total: {n}
    Grid points: {grid_points}""",
    xref="paper", yref="paper",
    x=1, y=1, showarrow=False
    )

st.plotly_chart(fig_pl)

st.text('Page 62 of the book "Statistical Rethinking 2ed" by Richard McElreath')
st.markdown("""We will call such simulated data dummy data, to indicate that it is a stand-in for actual
data. With the globe tossing model, the dummy data arises from a binomial likelihood:""")
st.latex(r'''
Pr(W \mid N, p) =  \frac{N!}{W!(N − W)!}p^W (1 − p)^{N−W}
''')
st.markdown("""where W is an observed count of “water” and N is the number of tosses.""")

df = pr_data_given_p(n, p)

fig2 = px.bar(df, x="k", y="prob")
# fig2.update_xaxes(range=[1.5, 4.5])
fig2.update_yaxes(range=[0, 1])
fig2.add_annotation(
    text=f"""
    Total: {n}
    Probability: {p}""",
    xref="paper", yref="paper",
    x=1, y=1, showarrow=False
    )
st.plotly_chart(fig2)
