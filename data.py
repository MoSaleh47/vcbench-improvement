# import pandas as pd

# pub  = pd.read_csv("vcbench_final_public.csv")
# priv = pd.read_csv("vcbench_final_private.csv")

# print("=== PUBLIC sample ===")
# print(repr(pub["anonymised_prose"].iloc[0][:300]))

# print("\n=== PRIVATE sample ===")
# print(repr(priv["anonymised_prose"].iloc[0][:300]))

# print("\n=== Private prose nulls ===")
# print(priv["anonymised_prose"].isna().sum(), "nulls out of", len(priv))

# print("\n=== Private prose empty strings ===")
# print((priv["anonymised_prose"].str.strip() == "").sum(), "empty strings")

# print("\n=== Private prose avg length ===")
# print(priv["anonymised_prose"].str.len().describe())

#%%
import pandas as pd

f =  pd.read_csv("results_private_test.csv")
print(f.head())

print(f["prediction"].value_counts())
#%%