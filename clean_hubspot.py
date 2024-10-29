import pandas as pd
import matplotlib.pyplot as plt
import scipy

df_original = pd.read_csv("all-deals.csv")
df_original = df_original[["Record ID","Amount", "Close Date", "Create Date"]] # 438 X 4
df_original = df_original.dropna() # 92 X 4

# Clean Create Date and Close Date 
df_original["Create Date"] = pd.to_datetime(df_original["Create Date"], errors="coerce")
df_original['Close Date'] = pd.to_datetime(df_original['Close Date'], errors='coerce')

# print(df_original.dtypes)  

# Check Record IDs - no issues all IDs are unique
duplicate_record_ids = df_original[df_original["Record ID"].duplicated()]
# print(duplicate_record_ids)

duplicate_count = df_original["Record ID"].duplicated().sum()
# print(f"Number of duplicate Record IDs: {duplicate_count}")


# Clean hubspot-crm-exports-all-deals-2024-08-27 no PII
hubspot_df = pd.read_csv("hubspot-crm-exports-all-deals-2024-08-27 no PII.csv")
hubspot_df = hubspot_df[["Record ID", "Deal Stage"]]
hubspot_df = hubspot_df.dropna()
# print(hubspot_df)

# Merge df_original and hubspot_df based on 'Record ID' column
merged_df = pd.merge(df_original, hubspot_df, on="Record ID", how="inner")
merged_df = merged_df.dropna()
# print(merged_df)


# Clean Deal Stage Column
deal_stage_mappings = {
    "Contract Signed ": "Contract Signed",
    "Contract Signed": "Contract Signed",
    "Contract Issued ": "Contract Issued",
    "Contract Issued": "Contract Issued",
    "Last Stage Meeting": "Last Stage Meeting",
    "Demo": "Demo",
    "Closed Lost ": "Closed Lost",
    "Closed Lost": "Closed Lost",
    "Paid ": "Paid",
    "Paid": "Paid",
    "Circle Back ": "Circle Back"
}

merged_df["Deal Stage"] = merged_df["Deal Stage"].replace(deal_stage_mappings)
print(merged_df["Deal Stage"].value_counts)

merged_df.to_csv("data/sales_info.csv")

# Summary Statistics

# Pie Chart: Amount Made per Deal Stage 
revenue_by_stage = merged_df.groupby("Deal Stage")["Amount"].sum()

plt.figure(figsize=(6, 8))
plt.pie(revenue_by_stage, labels=revenue_by_stage.index, autopct='%1.1f%%', startangle=90)
plt.title("University Statups mount Made by Deal Stage")
plt.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
plt.show()

# Bar Chart on the Number of Deals per Deal Stage
deals_by_stage = merged_df["Deal Stage"].value_counts()

plt.figure(figsize=(10, 6))
deals_by_stage.plot(kind='bar', color='skyblue')
plt.title("Number of Deals by Stage")
plt.xlabel("Deal Stage")
plt.ylabel("Number of Deals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 

