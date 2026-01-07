import streamlit as st

from ui_common import APP_TITLE, get_db_conn

st.set_page_config(page_title=APP_TITLE, layout="wide")

if "conn" not in st.session_state:
    st.session_state["conn"] = get_db_conn()

if hasattr(st, "navigation") and hasattr(st, "Page"):
    pages = [
        st.Page("home.py", title="Home", icon="🏠"),
        st.Page("pages/1_lookup_shape.py", title="Lookup Shape", icon="🔎"),
        st.Page("pages/2_fast_shapes.py", title="Fast Shapes", icon="🚀"),
        st.Page("pages/3_scaling_curves.py", title="Scaling Curves", icon="📈"),
        st.Page("pages/4_compare_hardware.py", title="Compare Hardware", icon="📊"),
        st.Page("pages/5_browse_export.py", title="Browse & Export", icon="📄"),
    ]
    st.navigation(pages).run()
else:
    st.title(APP_TITLE)
    st.info(
        "Your Streamlit version doesn't support `st.navigation`. "
        "Upgrade Streamlit or rename `app.py` and files in `pages/` to get nicer sidebar names."
    )
