import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# col: "Record ID","Deal Name","Last Activity Date","State","Deal Stage","Close Date","Deal owner","Amount","Date of last meeting booked in meetings tool","Pipeline","Associated Product","Products","LMS","Associated Product IDs"

all_deals = pd.read_csv("hubspot-crm-exports-all-deals-2024-08-27.csv")
# Dropped Close Date -- no data
all_deals = all_deals[["Record ID","Deal Name","Last Activity Date","State","Deal owner","Amount","Date of last meeting booked in meetings tool","Pipeline","Associated Product","Products","LMS","Associated Product IDs"]]
print(all_deals) # 358 X 13

all_deals_no_pii = pd.read_csv("sales_info.csv")
# Convert relevant columns to datetime
date_columns = ["Last Activity Date", "Date of last meeting booked in meetings tool"]
for col in date_columns:
    all_deals[col] = pd.to_datetime(all_deals[col], errors='coerce')

# Use 'left' join to keep all records in all_deals, even if they don't match in merged_df

all_deals_merged = pd.merge(all_deals, all_deals_no_pii[['Record ID', 'Deal Duration', 'Create Date', 'Close Date', 'Deal Stage']], 
    on='Record ID', how='inner'
)

all_deals_merged['Last Activity Date'] = pd.to_datetime(all_deals_merged['Last Activity Date'], errors='coerce')
all_deals_merged['Date of last meeting booked in meetings tool'] = pd.to_datetime(all_deals_merged['Date of last meeting booked in meetings tool'], errors='coerce')

# print(all_deals_merged.columns)

# Bar Chart of Average Duration by Deal Stage
average_duration_by_stage = all_deals_merged.groupby('Deal Stage')['Deal Duration'].mean().sort_values(ascending=False)
print("Average Duration by Deal Stage:")
print(average_duration_by_stage)

# Group by 'Deal Stage' and calculate the average duration
average_duration_by_stage = all_deals_merged.groupby('Deal Stage')['Deal Duration'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=average_duration_by_stage.values, y=average_duration_by_stage.index, palette='viridis')
plt.title('Average Deal Duration by Deal Stage')
plt.xlabel('Average Duration (days)')
plt.ylabel('Deal Stage')
plt.show()

# Display all unique values in the 'Deal Owner' column
unique_products = all_deals_merged['Associated Product'].unique()
print("Unique Products:")
print(unique_products)


# 1. Analyze Deal Duration by Deal Owner
# Calculate average deal duration by deal owner
average_duration_by_owner = all_deals_merged.groupby('Deal owner')['Deal Duration'].mean().sort_values(ascending=False)
print("Average Deal Duration by Deal Owner:")
print(average_duration_by_owner)

# Plot average duration by deal owner
plt.figure(figsize=(12, 6))
sns.barplot(x=average_duration_by_owner.index, y=average_duration_by_owner.values, palette='plasma')
plt.xticks(rotation=45)
plt.title('Average Deal Duration by Deal Owner')
plt.xlabel('Deal Owner')
plt.ylabel('Average Duration (days)')
plt.show()

# 2. Analyze Deal Duration by Deal Amount
# # Group deals by amount (you may want to bin amounts if there is a large range)
# all_deals_merged['Amount Binned'] = pd.qcut(all_deals_merged['Amount'], q=4, labels=["Low", "Medium", "High", "Very High"])
# average_duration_by_amount = all_deals_merged.groupby('Amount Binned')['Deal Duration'].mean().sort_values(ascending=False)
# print("Average Deal Duration by Deal Amount:")
# print(average_duration_by_amount)

# # Plot average duration by deal amount bins
# plt.figure(figsize=(10, 6))
# sns.barplot(x=average_duration_by_amount.index, y=average_duration_by_amount.values, palette='viridis')
# plt.title('Average Deal Duration by Deal Amount')
# plt.xlabel('Deal Amount Category')
# plt.ylabel('Average Duration (days)')
# plt.show()
# Calculate the quartiles and ranges for the "Amount" bins
amount_bins = pd.qcut(all_deals_merged['Amount'], q=4)
amount_ranges = amount_bins.cat.categories

# Add labels with ranges for each bin
amount_labels = [
    f"Low ({amount_ranges[0].left:.2f} - {amount_ranges[0].right:.2f})",
    f"Medium ({amount_ranges[1].left:.2f} - {amount_ranges[1].right:.2f})",
    f"High ({amount_ranges[2].left:.2f} - {amount_ranges[2].right:.2f})",
    f"Very High ({amount_ranges[3].left:.2f} - {amount_ranges[3].right:.2f})"
]

# Bin amounts using the custom labels
all_deals_merged['Amount Binned'] = pd.qcut(all_deals_merged['Amount'], q=4, labels=amount_labels)

# Calculate average deal duration by amount bin
average_duration_by_amount = (
    all_deals_merged.groupby('Amount Binned')['Deal Duration']
    .mean()
    .sort_values(ascending=False)
)

# Print average deal duration by amount category
print("Average Deal Duration by Deal Amount Range:")
print(average_duration_by_amount)

# Plot average duration by deal amount bins with ranges
plt.figure(figsize=(10, 6))
sns.barplot(x=average_duration_by_amount.index, y=average_duration_by_amount.values, palette='viridis')
plt.title('Average Deal Duration by Deal Amount Range')
plt.xlabel('Deal Amount Category (Range)')
plt.ylabel('Average Duration (days)')
plt.xticks(rotation=45)
plt.show()

# Drop rows with NaN in the 'Associated Product' column
all_deals_merged = all_deals_merged.dropna(subset=['Associated Product'])

# Group by 'Associated Product' and calculate the average duration
average_duration_by_product = all_deals_merged.groupby('Associated Product')['Deal Duration'].mean().sort_values(ascending=False)

# Display the average duration by product
print("Average Deal Duration by Associated Product:")
print(average_duration_by_product)

# Plot average duration by associated product
plt.figure(figsize=(12, 6))
sns.barplot(x=average_duration_by_product.index, y=average_duration_by_product.values, palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Average Deal Duration by Associated Product')
plt.xlabel('Associated Product')
plt.ylabel('Average Duration (days)')
plt.show()


###############################################################################
# Ensure Deal Duration is numeric
all_deals_merged['Deal Duration'] = pd.to_numeric(all_deals_merged['Deal Duration'], errors='coerce')

# Drop rows with missing Pipeline or Deal Duration values
all_deals_merged = all_deals_merged.dropna(subset=['Pipeline', 'Deal Duration'])
print(all_deals_merged)

# 1. Pipeline Comparison - Average Deal Duration by Pipeline
average_duration_by_pipeline = all_deals_merged.groupby('Pipeline')['Deal Duration'].mean().sort_values(ascending=False)
print("Average Deal Duration by Pipeline:")
print(average_duration_by_pipeline)

# Plot average duration by pipeline
plt.figure(figsize=(10, 6))
sns.barplot(x=average_duration_by_pipeline.index, y=average_duration_by_pipeline.values, palette='viridis')
plt.title('Average Deal Duration by Pipeline')
plt.xlabel('Pipeline')
plt.ylabel('Average Duration (days)')
plt.show()

# 2. Stage-by-Stage Analysis per Pipeline - Average Duration per Stage within Each Pipeline
stage_duration_by_pipeline = all_deals_merged.groupby(['Pipeline', 'Deal Stage'])['Deal Duration'].mean().unstack()
print("Average Deal Duration by Stage and Pipeline:")
print(stage_duration_by_pipeline)

# Plot stage duration by pipeline
plt.figure(figsize=(12, 8))
sns.heatmap(stage_duration_by_pipeline, annot=True, cmap='rocket', fmt=".1f", cbar_kws={'label': 'Avg Duration (days)'})
plt.title('Average Deal Duration by Stage and Pipeline')
plt.xlabel('Deal Stage')
plt.ylabel('Pipeline')
plt.show()






################################################################################
#Predictive Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prepare the data
all_deals_merged['Close Within 90 Days'] = all_deals_merged['Deal Duration'] <= 90
model_data = all_deals_merged[['Deal Stage', 'Pipeline', 'Amount', 'Associated Product', 'Last Activity Date', 'Close Within 90 Days']].dropna()

# Convert categorical columns to numerical (one-hot encoding)
model_data = pd.get_dummies(model_data, columns=['Deal Stage', 'Pipeline', 'Associated Product'])

# Split into features (X) and target (y)
X = model_data.drop(columns=['Close Within 90 Days', 'Last Activity Date'])
y = model_data['Close Within 90 Days']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Calculate probabilities for each stage to close within 30 and 90 days
all_deals_merged['Close Within 30 Days'] = all_deals_merged['Deal Duration'] <= 30
all_deals_merged['Close Within 90 Days'] = all_deals_merged['Deal Duration'] <= 90

# Group by stage and calculate probabilities
probability_by_stage = all_deals_merged.groupby('Deal Stage').agg({
    'Close Within 30 Days': 'mean',
    'Close Within 90 Days': 'mean'
}).sort_values(by='Close Within 90 Days', ascending=False)

print("Stage-by-Stage Probability of Closure within 30 and 90 Days:")
print(probability_by_stage)

# Plot probabilities by stage
plt.figure(figsize=(12, 6))
probability_by_stage.plot(kind='bar', color=['#FF7F0E', '#1F77B4'])
plt.title('Probability of Closing within 30 and 90 Days by Deal Stage')
plt.xlabel('Deal Stage')
plt.ylabel('Probability')
plt.legend(['Close within 30 Days', 'Close within 90 Days'])
plt.show()
