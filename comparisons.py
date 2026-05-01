import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load results
df = pd.read_csv("stats.csv")


#Accuracy box plot
plt.figure()
sns.barplot(data=df, x="size", y="accuracy", hue="model")

plt.title("Model Accuarcy Across Dataset Sizes")
plt.xlabel("Dataset Size")
plt.ylabel("Accuracy")
plt.legend(title="Model")

plt.tight_layout()
plt.show()

#F1 Score (fake class)
plt.figure()
sns.barplot(data=df, x="size", y="f1_0", hue="model")

plt.title("F1 Score (Fake Class) Across Dataset Sizes")
plt.xlabel("Dataset Size")
plt.ylabel("F1 Score (Fake)")
plt.legend(title="Model")

plt.tight_layout()
plt.show()

#Training time comparison
plt.figure()
sns.barplot(data=df, x="size", y="train_time", hue="model")

plt.title("Training Time Comparison")
plt.xlabel("Dataset Size")
plt.ylabel("Time (seconds)")

plt.tight_layout()
plt.show()

#Accuracy vs training time
plt.figure()
sns.scatterplot(
    data=df,
    x="train_time",
    y="accuracy",
    hue="model",
    s=120
)

for i in range(len(df)):
    plt.text(df["train_time"][i], df["accuracy"][i], df["size"][i])

plt.title("Accuracy vs Training Time Trade-off")
plt.xlabel("Training Time (s)")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()