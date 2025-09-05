import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate fake training metrics
epochs = list(range(1, 11))
accuracy = [0.60, 0.65, 0.68, 0.70, 0.73, 0.75, 0.78, 0.80, 0.82, 0.85]

df = pd.DataFrame({
    "epoch": epochs,
    "accuracy": accuracy
})

# Save metrics as CSV
df.to_csv("metrics.csv", index=False)

# 2. Plot accuracy
plt.figure(figsize=(6,4))
plt.plot(epochs, accuracy, marker="o", label="Accuracy")
plt.title("Model Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy.png")

print("Report generated: metrics.csv and accuracy.png")
