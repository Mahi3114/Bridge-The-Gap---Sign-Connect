import torch

def inspect_model(model_path):
    print(f"Loading PyTorch model from: {model_path}...\n")
    try:
        # Load the model on CPU to avoid any GPU mismatch issues
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        print("================ MODEL ARCHITECTURE ================")
        print(model)
        print("====================================================\n")
        
        print("HOW TO READ THIS:")
        print("1. Look at the VERY FIRST layer printed above.")
        print("2. Find the number next to 'in_features=' or 'input_size='.")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    # Using the exact path you provided!
    inspect_model(r'E:\SignConnect\new_model\best_model.pt')