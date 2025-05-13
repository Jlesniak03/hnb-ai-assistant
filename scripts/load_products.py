import pandas as pd


# Load your product spreadsheet
df = pd.read_excel("Productsv4.0.xlsx")  # make sure this file is in the same folder


# Strip extra whitespace from column names
df.columns = df.columns.str.strip()


# Check the cleaned column names
print(df.columns)


# Combine all fields into one searchable text per row
df["combined"] = df.apply(lambda row: f"""
Product Name: {row['Product Name']}
Feature: {row['Feature']}
Action: {row['Action']}
Benefit: {row['Benefit']}
Further Info: {row['Further info']}
Cautions: {row['Cautions']}
""", axis=1)


# Save the combined data to a new list
documents = df["combined"].tolist()


# Save all combined documents to a text file
with open("product_documents.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.strip() + "\n\n")


print("✅ Product data exported successfully.")


