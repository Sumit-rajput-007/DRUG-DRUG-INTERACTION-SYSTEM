import pandas as pd

# Load your dataset
df = pd.read_csv("db_drug_interactions.csv")

# Rename columns
df = df.rename(columns={
    "Drug 1": "Drug1",
    "Drug 2": "Drug2",
    "Interaction Description": "Description"
})

# Function to assign severity based on description
def assign_severity(desc):
    desc = str(desc).lower()

    if "cardiotoxic" in desc or "severe" in desc:
        return "High"
    elif "risk" in desc or "increase" in desc:
        return "Moderate"
    else:
        return "Low"

# Add Severity column
df["Severity"] = df["Description"].apply(assign_severity)

# Reorder columns
df = df[["Drug1", "Drug2", "Severity", "Description"]]

# Save new file
df.to_csv("drug_interactions.csv", index=False)

print("✅ Dataset converted successfully!")









