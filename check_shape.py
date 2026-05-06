import torch

# Load the weights
state_dict = torch.load(r'E:\SignConnect\new_model\best_model.pt', map_location='cpu')

# The second number in the shape of the first layer's weights is your exact input size
input_size = state_dict['lstm.weight_ih_l0'].shape[1]

print("\n" + "="*40)
print(f"YOUR EXACT INPUT SIZE IS: {input_size}")
print("="*40 + "\n")

if input_size == 258:
    print("Diagnosis: You did NOT use Face Landmarks (Pose + Left Hand + Right Hand).")
elif input_size == 1662:
    print("Diagnosis: You DID use Face Landmarks.")
else:
    print(f"Diagnosis: You used a custom setup of {input_size} keypoints.")

print("\nLayer shapes for the rebuild:")
for key in ['lstm.weight_ih_l0', 'fc1.weight', 'fc2.weight']:
    if key in state_dict:
        print(f"{key}: {state_dict[key].shape}")