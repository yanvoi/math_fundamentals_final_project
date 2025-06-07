# 🧠 Naive Bayes Discrete Classifier

This project provides an interactive **Streamlit app** for training and testing a custom **Naive Bayes classifier** on datasets with **categorical (discrete)** features.

---

## 🚀 Features

* 📂 Upload your own CSV file with categorical data
* ✏️ Edit the dataset directly in the browser (add, delete, fix values)
* 🎯 Choose your **target column** and **feature columns**
* 🧮 Optionally define **custom class priors**
* 🧠 Train a **Naive Bayes classifier**
* 📊 Visualize results with:

  * Posterior probability heatmap
  * Class distribution chart
  * Accuracy per class
  * Confusion matrix
  * Misclassified samples

---

## 🛠️ How It Works

At the core is the `NaiveBayesDiscrete` class that:

* Computes prior and conditional probabilities from the training data
* Uses **Bayes’ theorem** with **Laplace-like smoothing** for prediction
* Outputs both predicted labels and class posterior probabilities

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit
pandas
numpy
seaborn
matplotlib
```

---

## ▶️ How to Run

Launch the Streamlit app with:

```bash
streamlit run app.py
```

> Replace `app.py` with your actual script name if different.

---

## 📁 Example Input

Example CSV format:

```csv
outlook,temperature,humidity,windy,play
sunny,hot,high,false,no
rainy,cool,normal,true,no
overcast,hot,high,false,yes
```

* `play` is the **target**
* The rest are **categorical features**

---

## 📈 Visual Output

After training the model, you'll get:

* **Predicted labels**
* **Posterior probabilities**
* **Confusion matrix**
* **Per-class accuracy**
* **Visual insights for model evaluation**

---

## 🧪 Use Case

Ideal for:

* Teaching machine learning basics
* Visualizing Naive Bayes decisions
* Quick prototyping with categorical datasets
* Exploring the impact of class imbalance or prior probability

---

## 📬 Feedback

If you'd like to suggest improvements or report bugs, feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---