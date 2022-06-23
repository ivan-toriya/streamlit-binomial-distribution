import pandas as pd
import numpy as np

from scipy.stats import binom

import streamlit as st
import plotly.express as px

k = st.sidebar.slider(label='Water samples (w)',
              min_value=1,
              max_value=10,
              value=6)

n = st.sidebar.slider(label='Total (Water + Land) (n)',
              min_value=1,
              max_value=10,
              value=9)

grid_point = st.sidebar.slider(
    label='grid_points (each point = p)',
    min_value=0,
    max_value=100,
    value=100
    )

st.latex(r'''
Pr(w \mid n, p) =  \frac{n!}{w!(n − w)!} p^w (1 − p)^{n−w}
''')
st.markdown('''
Read the above as:

The counts of “water” W and “land’ L are distributed binomially, with probability p of “water” on each toss.
''')

@st.cache(persist=True, show_spinner=True)
def grid_approx(k, n, grid_point):
    # define grid
    p_grid = np.linspace(0, 1, grid_point)

    binom_distribution = binom.pmf(k, n=n, p=p_grid)

    df = pd.DataFrame(data={'prob_p': p_grid, 'prob_w': binom_distribution})
    
    return df

df = grid_approx(k, n, grid_point)

fig = px.area(df, x="prob_p", y="prob_w")

st.plotly_chart(fig)