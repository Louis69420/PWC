# Project context
Timed 4-hour PwC graduate assessment. Public repo, assessors will read everything
including this file. Six deliverables: framing, EDA, solution design, model,
Streamlit dashboard, 3-5 min video.

# Working rules
- Simplest working solution first; iterate only if I ask.
- Pin to: pandas, scikit-learn, matplotlib/seaborn, streamlit. No new deps
  without asking me.
- Every model-related suggestion must state which columns could leak
  (unavailable at prediction time).
- Remind me to use imbalance-aware metrics (precision/recall, PR-AUC) if working with imbalanced datasets and classification. Never report bare
  accuracy in this case.
- random_state=42 everywhere. Code must run top-to-bottom in a fresh kernel.
- Keep comments and text minimal but do not compromise readability.
- End each notebook section with a 1-2 sentence markdown takeaway tied to the business question.
- Prompt me to commit at each milestone.
- Leave everything to me in the README.md and don't change it in any way

# Structure
- notebooks/eda.ipynb, notebooks/model.ipynb
- dashboard/app.py (Streamlit), dashboard/screenshots/
- outputs/scored_records.csv feeds the dashboard
- README.md (assessor front door)

