import os
import shutil

# ─── CONFIG ───────────────────────────────────────────────
ISL_DATASET  = r"E:\SignConnect\ISL_DATASET"
MY_DATASET   = r"E:\SignConnect\my_DATASET"
# ──────────────────────────────────────────────────────────

DELETE_SIGNS = [
    "dog", "dress", "friday", "minute", "mouse", "old", "pant",
    "saturday", "shoes", "slow", "thursday", "tuesday",
    "warm", "wednesday", "week", "young", "Man", "doing", "second"
]

print("Deleting weak signs from both datasets...\n")

for sign in DELETE_SIGNS:
    # Delete from ISL_DATASET
    isl_path = os.path.join(ISL_DATASET, sign)
    if os.path.exists(isl_path):
        shutil.rmtree(isl_path)
        print(f"  ✅ Deleted from ISL_DATASET : {sign}")
    else:
        print(f"  ⚠️  Not found in ISL_DATASET: {sign}")

    # Delete from my_DATASET
    my_path = os.path.join(MY_DATASET, sign)
    if os.path.exists(my_path):
        shutil.rmtree(my_path)
        print(f"  ✅ Deleted from my_DATASET  : {sign}")
    else:
        print(f"  ⚠️  Not found in my_DATASET : {sign}")

    print()

print("🎉 Cleanup done!")
print("\nRemaining signs in ISL_DATASET:")
remaining = sorted(os.listdir(ISL_DATASET))
for s in remaining:
    print(f"  {s}")
print(f"\nTotal remaining: {len(remaining)} signs")