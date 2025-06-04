import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# Naive Bayes Classifier
class NaiveBayesDiscrete:
    def __init__(self, class_priors=None):
        self.class_priors = class_priors
        self.likelihoods = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.likelihoods = {c: defaultdict(lambda: defaultdict(int)) for c in self.classes}

        if self.class_priors is None:
            counts = np.unique(y, return_counts=True)
            self.class_priors = {cls: cnt / len(y) for cls, cnt in zip(*counts)}

        for xi, label in zip(X, y):
            for idx, val in enumerate(xi):
                self.likelihoods[label][idx][val] += 1

        for c in self.classes:
            for idx in self.likelihoods[c]:
                total = sum(self.likelihoods[c][idx].values())
                for val in self.likelihoods[c][idx]:
                    self.likelihoods[c][idx][val] /= total

    def predict(self, X):
        predictions = []
        posteriors = []

        for xi in X:
            class_probs = {}
            for c in self.classes:
                prob = self.class_priors.get(c, 1e-6)
                for idx, val in enumerate(xi):
                    prob *= self.likelihoods[c][idx].get(val, 1e-6)
                class_probs[c] = prob

            total = sum(class_probs.values())
            class_posteriors = {c: class_probs[c] / total for c in class_probs}
            predicted_class = max(class_posteriors, key=class_posteriors.get)
            predictions.append(predicted_class)
            posteriors.append(class_posteriors)

        return np.array(predictions), posteriors

# --- Streamlit UI ---
st.set_page_config(page_title="Naive Bayes Discrete Classifier", layout="centered")

st.title("🧠 Naive Bayes Discrete Classifier Demo")
st.markdown("Upload a categorical dataset and see predictions in action!")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("📝 **Edit Your Dataset (Optional)**")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    st.caption("You can directly edit cells, add rows, or fix typos here.")

    if st.checkbox("⬇️ Allow download of edited CSV"):
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Edited CSV", data=csv, file_name="edited_dataset.csv", mime='text/csv')

    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 Select Target Column", columns)
    feature_cols = st.multiselect("🧩 Select Feature Columns", [col for col in columns if col != target_col], default=[col for col in columns if col != target_col])

    if st.button("Train & Predict"):
        X = edited_df[feature_cols].astype(str).values
        y = edited_df[target_col].astype(str).values

        # Optional custom priors
        unique_classes = np.unique(y)
        st.markdown("🧮 Customize Class Priors (Optional)")
        class_priors = {}
        total_weight = 0
        for cls in unique_classes:
            val = st.number_input(f"Prior for class '{cls}'", min_value=0.0, max_value=1.0, value=1.0 / len(unique_classes), step=0.01)
            class_priors[cls] = val
            total_weight += val
        class_priors = {cls: val / total_weight for cls, val in class_priors.items()}

        model = NaiveBayesDiscrete(class_priors)
        model.fit(X, y)
        preds, posteriors = model.predict(X)

        results_df = pd.DataFrame(posteriors)
        results_df['True Label'] = y
        results_df['Predicted Label'] = preds
        results_df.index.name = 'Sample'

        st.success("✅ Prediction Complete!")
        st.write("📊 Results Table")
        st.dataframe(results_df)

        # Heatmap
        st.markdown("📈 Posterior Probability Heatmap")
        fig, ax = plt.subplots(figsize=(10, len(results_df)*0.5))
        sns.heatmap(results_df.drop(columns=['True Label', 'Predicted Label']), annot=True, cmap='YlGnBu', ax=ax)
        plt.xlabel("Class")
        plt.ylabel("Sample Index")
        st.pyplot(fig)

        # Misclassification highlight
        wrong_preds = results_df[results_df['True Label'] != results_df['Predicted Label']]
        if not wrong_preds.empty:
            st.warning("⚠️ Misclassified Samples")
            st.dataframe(wrong_preds)

        st.markdown("💡 Tip: Try changing class priors or modifying the dataset to explore model behavior.")

else:
    st.info("📂 Please upload a CSV file with **only categorical** features.")
