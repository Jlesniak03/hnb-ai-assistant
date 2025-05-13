import pandas as pd 

# Load your ailments spreadsheet
df = pd.read_excel("./Data/Ailmentsv4.0.xlsx")

# Clean the column names
df.columns = df.columns.str.strip()

# Show all available columns to confirm
print("Available columns:", df.columns.tolist())

# Combine all fields into one searchable text per row
df["combined"] = df.apply(lambda row: f"""
Ailment: {row['Ailment']}
Symptoms: {row['Symptoms']}
Relief: {row['relief']}
Prevention: {row['prevention']}
Complementary: {row['complementary']}
Further Info: {row['Further info']}
""", axis=1)

# Save the combined data to a new list
documents = df["combined"].tolist()

# Save all combined documents to a text file (one entry per line)
with open("./Data/ailment_documents.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.strip() + "\n\n")

print("✅ Ailment data exported successfully.")

