"""GameVault Acquisition Radar - Streamlit dashboard.

Reads only from outputs/ (produced by the notebooks), so it runs from a clone
of the public repo without the raw client dataset.
Run: streamlit run dashboard/app.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='GameVault Acquisition Radar', page_icon='🎯', layout='wide')

OUT = Path(__file__).resolve().parent.parent / 'outputs'
FEATURES = ['review', 'hour_played', 'is_early_access_review']


@st.cache_data
def load_data():
    scorecard = pd.read_csv(OUT / 'game_scorecard.csv')
    records = pd.read_csv(OUT / 'scored_records.csv', parse_dates=['date_posted'])
    return scorecard, records


@st.cache_resource
def load_model():
    with open(OUT / 'sentiment_model.pkl', 'rb') as f:
        return pickle.load(f)


scorecard, records = load_data()
model = load_model()

# ---------------- sidebar: strategy controls ----------------
st.sidebar.title('🎯 Acquisition Radar')
st.sidebar.caption('GameVault Publishing — find lasting appeal, not temporary buzz.')

st.sidebar.subheader('Scorecard weights')
st.sidebar.caption('The components are the analysis; the weighting is strategy. Tune it live.')
w_sent = st.sidebar.slider('Sentiment health', 0.0, 1.0, 0.35, 0.05)
w_mom = st.sidebar.slider('Momentum (lasting appeal)', 0.0, 1.0, 0.25, 0.05)
w_eng = st.sidebar.slider('Engagement depth', 0.0, 1.0, 0.25, 0.05)
w_com = st.sidebar.slider('Community validation', 0.0, 1.0, 0.15, 0.05)
hide_thin = st.sidebar.toggle('Hide titles with <100 reviews', value=False)

weights = np.array([w_sent, w_mom, w_eng, w_com])
weights = weights / weights.sum() if weights.sum() else np.array([.35, .25, .25, .15])

sc = scorecard.copy()
sc['composite'] = (sc[['score_sentiment', 'score_momentum',
                       'score_engagement', 'score_community']] @ weights) * 100
sc = sc.sort_values('composite', ascending=False).reset_index(drop=True)
sc.index += 1  # rank
if hide_thin:
    sc = sc[~sc['thin_coverage']]

tab_board, tab_dive, tab_scorer = st.tabs(
    ['🏆 Leaderboard', '🔍 Game deep-dive', '⚡ Live review scorer'])

# ---------------- tab 1: leaderboard ----------------
with tab_board:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Titles tracked', f"{scorecard.shape[0]}")
    k2.metric('Reviews analysed', f"{len(records):,}")
    k3.metric('Positive overall', f"{(records['recommendation'] == 'Recommended').mean():.0%}")
    k4.metric('Engine PR-AUC (unseen games)', '0.75')
    st.caption('Engine quality verified in `notebooks/02_model.ipynb` (GroupKFold on titles the model never trained on).')

    view = sc[['title', 'composite', 'n_reviews', 'pct_recommended', 'momentum',
               'median_hours', 'thin_coverage']].copy()
    view['confidence'] = np.where(view.pop('thin_coverage'), '⚠️ thin data', '✅ adequate')
    st.dataframe(
        view,
        width='stretch',
        column_config={
            'title': 'Game',
            'composite': st.column_config.ProgressColumn(
                'Acquisition score', min_value=0, max_value=100, format='%.0f'),
            'n_reviews': st.column_config.NumberColumn('Reviews'),
            'pct_recommended': st.column_config.NumberColumn('Positive', format='percent'),
            'momentum': st.column_config.NumberColumn('Momentum (6m vs life)', format='percent'),
            'median_hours': st.column_config.NumberColumn('Median hours'),
            'confidence': 'Confidence',
        },
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader('Sentiment vs engagement')
        st.caption('Top-right = loved and deeply played. Bubble size = review volume.')
        st.scatter_chart(sc, x='median_hours', y='pct_recommended',
                         size='n_reviews', color='composite', height=380)
    with right:
        st.subheader('Tracked shortlist')
        st.caption('Pursue/pass in one place — no more scattered spreadsheets.')
        shortlist = st.multiselect('Shortlist for partnership talks',
                                   sc['title'].tolist(),
                                   default=sc['title'].head(3).tolist())
        if shortlist:
            st.dataframe(
                sc[sc['title'].isin(shortlist)][['title', 'composite', 'pct_recommended', 'last_review']],
                width='stretch', hide_index=True)

# ---------------- tab 2: deep-dive ----------------
with tab_dive:
    game = st.selectbox('Choose a title (ordered by acquisition score)', sc['title'].tolist())
    row = sc[sc['title'] == game].iloc[0]
    g = records[records['title'] == game].sort_values('date_posted')

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('Acquisition score', f"{row['composite']:.0f}/100")
    k2.metric('Positive (lifetime)', f"{row['pct_recommended']:.0%}")
    k3.metric('Momentum', f"{row['momentum']:+.1%}")
    k4.metric('Median hours', f"{row['median_hours']:.0f}h")
    k5.metric('Reviews', f"{int(row['n_reviews']):,}")
    if row['thin_coverage']:
        st.warning('Under 100 reviews — treat every number on this page as indicative only.')

    q = (g.assign(rec=(g['recommendation'] == 'Recommended').astype(int))
          .set_index('date_posted').resample('QS')['rec'].agg(['mean', 'count']))
    q = q[q['count'] > 0]
    trend, vol = st.columns(2)
    with trend:
        st.subheader('Sentiment trajectory (quarterly)')
        st.caption('Crashes here are the risk a lifetime average hides.')
        st.line_chart((q['mean'] * 100).rename('% positive'), height=260)
    with vol:
        st.subheader('Review volume (quarterly)')
        st.caption('Sustained volume = sustained attention.')
        st.bar_chart(q['count'].rename('reviews'), height=260)

    st.subheader('What players are saying')
    voice = st.radio('Filter', ['Most recent', 'Most negative', 'Most positive'], horizontal=True)
    if voice == 'Most negative':
        sample = g.nsmallest(8, 'sentiment')
    elif voice == 'Most positive':
        sample = g.nlargest(8, 'sentiment')
    else:
        sample = g.tail(8).iloc[::-1]
    st.dataframe(
        sample[['date_posted', 'sentiment', 'recommendation', 'hour_played', 'review_snippet']],
        width='stretch', hide_index=True,
        column_config={
            'date_posted': st.column_config.DateColumn('Posted'),
            'sentiment': st.column_config.ProgressColumn('Engine sentiment', min_value=0, max_value=1, format='%.2f'),
            'recommendation': 'Player verdict',
            'hour_played': st.column_config.NumberColumn('Hours'),
            'review_snippet': st.column_config.TextColumn('Review (snippet)', width='large'),
        })

# ---------------- tab 3: live scorer ----------------
with tab_scorer:
    st.subheader('Score any review text — from any source')
    st.caption('This is how GameVault evaluates feedback that has no Steam label: '
               'forums, other storefronts, console communities.')
    text = st.text_area('Paste a review', height=140,
                        placeholder='e.g. "Loved the first 50 hours, but the servers crash constantly since the last update..."')
    c1, c2 = st.columns(2)
    hours = c1.number_input('Hours played before reviewing (if known)', 0, 20000, 20)
    ea = c2.checkbox('Posted during early access')

    if st.button('Score it', type='primary') and text.strip():
        x = pd.DataFrame([[text, hours, ea]], columns=FEATURES)
        p = model.predict_proba(x)[0, 1]
        verdict = 'POSITIVE' if p >= 0.5 else 'NEGATIVE'
        st.metric('Engine sentiment', f'{p:.0%}', verdict)
        st.progress(float(p))

        # per-word contributions: tf-idf values x model coefficients for this text
        pre, clf = model.named_steps['pre'], model.named_steps['clf']
        vec = pre.named_transformers_['text']
        text_mask = pd.Index(pre.get_feature_names_out()).str.startswith('text__')
        contrib = vec.transform([text]).multiply(clf.coef_[0][text_mask]).tocoo()
        terms = pd.Series(contrib.data, index=vec.get_feature_names_out()[contrib.col])
        if not terms.empty:
            pos, neg = st.columns(2)
            with pos:
                st.markdown('**Pushing positive**')
                for t, v in terms.nlargest(5).items():
                    if v > 0:
                        st.markdown(f'🟢 `{t}` (+{v:.2f})')
            with neg:
                st.markdown('**Pushing negative**')
                for t, v in terms.nsmallest(5).items():
                    if v < 0:
                        st.markdown(f'🔴 `{t}` ({v:.2f})')
        else:
            st.info('No words the engine recognises — score driven by hours/early-access only.')
