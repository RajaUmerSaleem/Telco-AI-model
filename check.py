import pandas as pd

# Load the dataset with proper parsing
df = pd.read_csv("telco_risk_levels.csv", sep=",")

# Print basic information
print(f"Number of instances: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# Print column names
print("\nColumn names:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found")

# Check for 'unknown' values which might be coded missing values
print("\nColumns with 'unknown' values:")
for col in df.columns:
    if df[col].dtype == 'object' and (df[col] == 'unknown').any():
        print(f"{col}: {(df[col] == 'unknown').sum()} unknown values")

# Check target variable 'y'
if 'y' in df.columns:
    print(f"\nNumber of unique classes in 'y': {df['y'].nunique()}")
    print(f"Classes: {df['y'].unique()}")
    
    # Show target distribution
    print("\nTarget distribution:")
    print(df['y'].value_counts())
else:
    print("\nTarget variable 'y' not found.")

# Show a few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Evaluate against project requirements
print("\n--- Project Requirements Evaluation ---")
meets_requirements = True

# Check for 4+ classes
if 'y' in df.columns and df['y'].nunique() < 4:
    print(f"❌ Has only {df['y'].nunique()} classes. Requirement: at least 4 classes")
    meets_requirements = False
else:
    print(f"✅ Classes requirement: {df['y'].nunique()} classes")

# Check for 2.5K+ instances
if df.shape[0] < 2500:
    print(f"❌ Has only {df.shape[0]} instances. Requirement: at least 2,500 instances")
    meets_requirements = False
else:
    print(f"✅ Instances requirement: {df.shape[0]} instances")

# Check for 20+ features
if df.shape[1] < 21:  # 20 features + 1 target
    print(f"❌ Has only {df.shape[1]-1} features. Requirement: at least 20 features")
    meets_requirements = False
else:
    print(f"✅ Features requirement: {df.shape[1]-1} features")

if meets_requirements:
    print("\nThis dataset meets all basic requirements.")
else:
    print("\nThis dataset does NOT meet all requirements for the project.")
    print("Consider finding another dataset from UCI or Kaggle that has:")
    print("- At least 4 classes in the target variable")
    print("- At least 2,500 instances")
    print("- At least 20 features")