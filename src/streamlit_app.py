#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import io
import joblib
import pandas as pd
from PIL import Image
import streamlit as st

# ---------- text cleaning ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- path helpers ----------
def project_root() -> Path:
    # this file is in src/, project root is parent directory
    return Path(__file__).resolve().parents[1]

def default_paths():
    root = project_root()
    out = root / "outputs"
    return {
        "pipeline": out / "pipeline.joblib",
        "model": out / "model.joblib",
        "vectorizer": out / "vectorizer.joblib",
    }

def load_pipeline_or_parts(pipeline_path: Path, model_path: Path, vectorizer_path: Path):
    """Attempt to load pipeline, model and vectorizer. Return a tuple
    (pipeline_or_none, clf_or_none, vec_or_none). Each component is loaded
    independently if the corresponding file exists.
    """
    pipe = None
    clf = None
    vec = None
    if pipeline_path and pipeline_path.exists():
        try:
            pipe = joblib.load(pipeline_path)
        except Exception:
            pipe = None
    if model_path.exists():
        try:
            clf = joblib.load(model_path)
        except Exception:
            clf = None
    if vectorizer_path.exists():
        try:
            vec = joblib.load(vectorizer_path)
        except Exception:
            vec = None
    return pipe, clf, vec


def detect_text_column(df: pd.DataFrame) -> str:
    candidates = [c for c in ["combined_text", "text", "content", "article", "body", "headline", "title"] if c in df.columns]
    if candidates:
        return candidates[0]
    # fallback: first object/string column
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            return c
    # otherwise just return the first column
    return df.columns[0]


def predict_texts(texts, pipe, clf, vec):
    if pipe is not None:
        probs = pipe.predict_proba(texts)[:, 1]
    else:
        X = vec.transform(texts)
        probs = clf.predict_proba(X)[:, 1]
    return probs

# ---------- streamlit app ----------
def main():
    # parse CLI overrides but give safe defaults relative to repo root
    dp = default_paths()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", default=str(dp["pipeline"]))
    ap.add_argument("--model", default=str(dp["model"]))
    ap.add_argument("--vectorizer", default=str(dp["vectorizer"]))
    args, _ = ap.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    model_path = Path(args.model).resolve()
    vectorizer_path = Path(args.vectorizer).resolve()

    st.set_page_config(page_title="Fake News Detector", layout="centered")
    st.title("Fake News & Misinformation Detector")
    st.caption("TF-IDF + RandomForest (TF-IDF pipeline) — batch predictions & charts")

    # sidebar: show where we look for files
    with st.sidebar:
        st.subheader("Model Artifacts")
        st.code(f"pipeline:  {pipeline_path}\nmodel:     {model_path}\nvectorizer:{vectorizer_path}")
        st.write(f"Exists → pipeline: **{pipeline_path.exists()}**, "
                 f"model: **{model_path.exists()}**, vectorizer: **{vectorizer_path.exists()}**")

    pipe, clf, vec = load_pipeline_or_parts(pipeline_path, model_path, vectorizer_path)
    if pipe is None and (clf is None or vec is None):
        st.error(
            "Model artifacts not found.\n\n"
            "• Ensure you trained and saved files to `outputs/`\n"
            "• Or run Streamlit with explicit paths, e.g.:\n"
            "  `streamlit run src/app.py -- --model C:/path/outputs/model.joblib --vectorizer C:/path/outputs/vectorizer.joblib`\n"
            "• From CLI, also try: `python -c \"import os; print(os.getcwd())\"` to see your working directory."
        )
        st.stop()

    # Model selection (if multiple artifact options available)
    model_options = []
    if pipeline_path.exists():
        model_options.append("pipeline")
    if model_path.exists() and vectorizer_path.exists():
        model_options.append("model+vectorizer")
    chosen = st.sidebar.selectbox("Model to use", model_options, index=0 if model_options else None)

    # Decision threshold
    threshold = st.sidebar.slider("FAKE decision threshold", 0.01, 0.99, 0.50, 0.01)

    st.header("Single text analysis")
    txt = st.text_area("Paste headline or article text:", height=160)
    if st.button("Analyze") and txt.strip():
        s = clean_text(txt)
        prob_fake = float(predict_texts([s], pipe if chosen == "pipeline" else None, clf if chosen == "model+vectorizer" else None, vec if chosen == "model+vectorizer" else None)[0])
        label = "FAKE" if prob_fake >= threshold else "REAL"
        st.metric("Prediction", label)
        st.progress(int(prob_fake * 100))
        st.write(f"Fake probability: {prob_fake:.1%} (threshold {threshold:.2f})")

    st.markdown("---")
    st.header("Batch predictions from CSV")
    uploaded = st.file_uploader("Upload a CSV file for batch predictions", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(io.StringIO(uploaded.getvalue().decode("utf-8")))

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        text_col_default = detect_text_column(df)
        text_col = st.selectbox("Text column to use for predictions", options=list(df.columns), index=list(df.columns).index(text_col_default))

        if st.button("Run batch predictions"):
            texts = df[text_col].fillna("").astype(str).map(clean_text).tolist()
            probs = predict_texts(texts, pipe if chosen == "pipeline" else None, clf if chosen == "model+vectorizer" else None, vec if chosen == "model+vectorizer" else None)
            df["fake_prob"] = probs
            df["predicted"] = df["fake_prob"].apply(lambda p: "FAKE" if p >= threshold else "REAL")
            st.write("Predictions (first 10):")
            st.dataframe(df[[text_col, "fake_prob", "predicted"]].head(10))
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    st.sidebar.markdown("---")
    st.sidebar.header("Evaluation charts")
    charts_dir = project_root() / "outputs" / "charts"
    if st.sidebar.checkbox("Show confusion / ROC / PR charts"):
        cm_path = charts_dir / "confusion_matrix.png"
        roc_path = charts_dir / "roc_curve.png"
        pr_path = charts_dir / "pr_curve.png"
        if cm_path.exists():
            st.subheader("Confusion Matrix")
            st.image(Image.open(cm_path), use_column_width=True)
        else:
            st.info("No confusion matrix found in outputs/charts/")
        if roc_path.exists():
            st.subheader("ROC Curve")
            st.image(Image.open(roc_path), use_column_width=True)
        if pr_path.exists():
            st.subheader("Precision-Recall Curve")
            st.image(Image.open(pr_path), use_column_width=True)

if __name__ == "__main__":
    main()
