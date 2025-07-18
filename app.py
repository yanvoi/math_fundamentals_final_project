import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any


class NaiveBayesDiscrete:
    """
    A Naive Bayes classifier for datasets with discrete (categorical) features.

    Attributes:
        class_priors (Optional[Dict[Any, float]]): Prior probabilities for each class.
        likelihoods (Dict[Any, Dict[int, Dict[str, float]]]): Likelihoods of feature values given class labels.
            Defaults to {}, which means class priors will be computed from the data.
        classes (List[Any]): List of unique class labels in the dataset.
            Defaults to [], which means class priors will be computed from the data.
    """
    def __init__(self, class_priors: Optional[Dict[Any, float]] = None):
        """Initialize the classifier.

        Args:
            class_priors (Optional[Dict[Any, float]]): Prior probabilities for each class.
        """
        self.class_priors = class_priors
        self.likelihoods: Dict[Any, Dict[int, Dict[str, float]]] = {}
        self.classes: List[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Naive Bayes classifier on training data.

        Args:
            X (np.ndarray): 2D array of features (categorical values as strings).
            y (np.ndarray): 1D array of class labels.
        """
        self.classes = np.unique(y)
        self.likelihoods = {c: defaultdict(lambda: defaultdict(int)) for c in self.classes}

        # If no class priors provided, compute them from data
        if self.class_priors is None:
            counts = np.unique(y, return_counts=True)
            self.class_priors = {cls: cnt / len(y) for cls, cnt in zip(*counts)}

        # Count feature value occurrences per class
        for xi, label in zip(X, y):
            for idx, val in enumerate(xi):
                self.likelihoods[label][idx][val] += 1

        # Normalize to get probabilities
        for c in self.classes:
            for idx in self.likelihoods[c]:
                total = sum(self.likelihoods[c][idx].values())
                for val in self.likelihoods[c][idx]:
                    self.likelihoods[c][idx][val] /= total

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict[Any, float]]]:
        """Predict the class labels for given data.

        Args:
            X (np.ndarray): 2D array of features to predict.

        Returns:
            Tuple[np.ndarray, List[Dict[Any, float]]]:
                - Predicted class labels.
                - Posterior probabilities for each class.
        """
        predictions = []
        posteriors = []

        for xi in X:
            class_probs: Dict[Any, float] = {}
            for c in self.classes:
                prob = self.class_priors.get(c, 1e-6)  # Fallback for unseen class
                for idx, val in enumerate(xi):
                    prob *= self.likelihoods[c][idx].get(val, 1e-6)  # Smoothing for unseen feature values
                class_probs[c] = prob

            total = sum(class_probs.values())
            class_posteriors = {c: class_probs[c] / total for c in class_probs}
            predicted_class = max(class_posteriors, key=class_posteriors.get)

            predictions.append(predicted_class)
            posteriors.append(class_posteriors)

        return np.array(predictions), posteriors


# Streamlit UI
st.set_page_config(page_title="Naive Bayes Discrete Classifier", layout="centered")

st.title("🧠 Naive Bayes Discrete Classifier Demo")
st.markdown("Upload a categorical dataset and see predictions in action!")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("📜 **Edit Your Dataset (Optional)**")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    st.caption("You can directly edit cells, add rows, or fix typos here.")

    # Optionally allow users to download the edited dataset
    if st.checkbox("⬇️ Allow download of edited CSV"):
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Edited CSV", data=csv, file_name="edited_dataset.csv", mime='text/csv')

    columns = edited_df.columns.tolist()
    target_col = st.selectbox("🎯 Select Target Column", columns)
    feature_cols = st.multiselect(
        "🧩 Select Feature Columns",
        [col for col in columns if col != target_col], default=[col for col in columns if col != target_col]
    )

    if target_col and feature_cols:
        y_vals = edited_df[target_col].astype(str).values
        unique_classes = np.unique(y_vals)

        # Set prior probabilities for each class manually
        st.markdown("🧮 Customize Class Priors (Optional)")
        for cls in unique_classes:
            st.number_input(
                f"Prior for class '{cls}'",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(unique_classes),
                step=0.01,
                key=f"prior_{cls}"
            )

        # Train the model and make predictions
        if st.button("Train & Predict"):
            X = edited_df[feature_cols].astype(str).values
            y = y_vals

            class_priors_raw = {cls: st.session_state[f"prior_{cls}"] for cls in unique_classes}
            total = sum(class_priors_raw.values())
            class_priors = {cls: val / total for cls, val in class_priors_raw.items()}

            model = NaiveBayesDiscrete(class_priors)
            model.fit(X, y)
            preds, posteriors = model.predict(X)

            results_df = pd.DataFrame(posteriors)
            results_df['True Label'] = y
            results_df['Predicted Label'] = preds
            results_df.index.name = 'Sample'

            # Save to session state for display below
            st.session_state["model"] = model
            st.session_state["results_df"] = results_df
            st.session_state["y"] = y

    # --- Display Results ---
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        y = st.session_state["y"]

        st.success("✅ Prediction Complete!")
        st.write("📊 Results Table")
        st.dataframe(results_df)

        # Posterior heatmap
        st.markdown("📈 Posterior Probability Heatmap")
        fig, ax = plt.subplots(figsize=(10, len(results_df) * 0.5))
        sns.heatmap(results_df.drop(columns=['True Label', 'Predicted Label']), annot=True, cmap='YlGnBu', ax=ax)
        plt.xlabel("Class")
        plt.ylabel("Sample Index")
        st.pyplot(fig)

        # Class distribution
        st.markdown("📊 Class Distribution in Dataset")
        class_counts = pd.Series(y).value_counts()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=class_counts.index, y=class_counts.values, palette='Set2', ax=ax1)
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        ax1.set_title("Original Class Distribution")
        st.pyplot(fig1)

        # Accuracy by class
        st.markdown("✅ Prediction Accuracy by Class")
        accuracy_by_class = results_df.groupby('True Label').apply(
            lambda g: (g['True Label'] == g['Predicted Label']).sum() / len(g)
        )
        fig2, ax2 = plt.subplots()
        sns.barplot(x=accuracy_by_class.index, y=accuracy_by_class.values, palette='Set1', ax=ax2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Per-Class Accuracy")
        st.pyplot(fig2)

        # Confusion matrix
        st.markdown("🔁 Confusion Matrix")
        confusion = pd.crosstab(
            results_df['True Label'],
            results_df['Predicted Label'],
            rownames=['Actual'],
            colnames=['Predicted'],
            normalize='index'
        )
        fig3, ax3 = plt.subplots()
        sns.heatmap(confusion, annot=True, fmt=".2f", cmap='Blues', ax=ax3)
        st.pyplot(fig3)

        # Show misclassified samples
        wrong_preds = results_df[results_df['True Label'] != results_df['Predicted Label']]
        if not wrong_preds.empty:
            st.warning("⚠️ Misclassified Samples")
            st.dataframe(wrong_preds)

        st.markdown("💡 Tip: Try changing class priors or modifying the dataset to explore model behavior.")

else:
    st.info("📂 Please upload a CSV file with **only categorical** features.")
