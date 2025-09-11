import numpy as np

# -------------------------------
# Step 1: Create dataset
# -------------------------------
data = np.array([
    [1, "Alice",   78, 85, 80],
    [2, "Bob",     55, 60, 58],
    [3, "Charlie", 89, 91, 86],
    [4, "David",   90, 88, 92],
    [5, "Eva",     67, 70, 72],
    [6, "Frank",   45, 50, 52],
    [7, "Grace",   82, 79, 85],
    [8, "Helen",   95, 94, 90]
], dtype=object)

# Separate columns for easier handling
names = data[:, 1]
maths = data[:, 2].astype(int)
science = data[:, 3].astype(int)
english = data[:, 4].astype(int)

# -------------------------------
# Step 2: Basic Loading & Viewing
# -------------------------------
print("Shape of dataset:", data.shape)

print("\nMaths column:", maths)

print("\nFirst 5 students:\n", data[:5])

print("\nLast 3 students:\n", data[-3:])

# -------------------------------
# Step 3: Statistics with NumPy
# -------------------------------
print("\nAverage marks - Maths:", np.mean(maths))
print("Average marks - Science:", np.mean(science))
print("Average marks - English:", np.mean(english))

print("\nHighest marks - Maths:", np.max(maths))
print("Highest marks - Science:", np.max(science))
print("Highest marks - English:", np.max(english))

print("\nLowest marks - Maths:", np.min(maths))
print("Lowest marks - Science:", np.min(science))
print("Lowest marks - English:", np.min(english))

# -------------------------------
# Step 4: Filtering & Conditions
# -------------------------------
print("\nStudents with Maths > 80:", names[maths > 80])
print("Students with Science < 60:", names[science < 60])
print("Students scoring > 75 in all subjects:", names[(maths > 75) & (science > 75) & (english > 75)])

# -------------------------------
# Step 5: Total & Rank
# -------------------------------
totals = maths + science + english
print("\nTotal marks of each student:", totals)

# Student with highest total
top_index = np.argmax(totals)
print("Topper:", names[top_index], "with", totals[top_index], "marks")

# Sort students by total marks
sorted_indices = np.argsort(-totals)   # descending
print("\nRanking (by total marks):")
for i, idx in enumerate(sorted_indices, start=1):
    print(i, "-", names[idx], totals[idx])
