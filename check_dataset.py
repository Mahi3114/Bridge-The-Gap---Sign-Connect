import os

my  = r"E:\SignConnect\my_DATASET"
isl = r"E:\SignConnect\ISL_DATASET"

signs = sorted(set(os.listdir(my)) | set(os.listdir(isl)))

print(f"{'Sign':<20} {'Original':>10} {'Yours':>10} {'Total':>10}")
print("-" * 52)

for sign in signs:
    orig = len([f for f in os.listdir(os.path.join(isl, sign))
                if f.lower().endswith(('.mov', '.mp4', '.avi'))])  \
           if os.path.exists(os.path.join(isl, sign)) else 0

    mine = len([f for f in os.listdir(os.path.join(my, sign))
                if f.endswith('.npy')]) \
           if os.path.exists(os.path.join(my, sign)) else 0

    print(f"{sign:<20} {orig:>10} {mine:>10} {orig+mine:>10}")

print("-" * 52)
print(f"{'TOTAL':<20} {sum(1 for _ in signs):>10}")