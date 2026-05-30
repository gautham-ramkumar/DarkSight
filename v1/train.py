import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from v1.model import FlatZero3DCE
from v1.losses import ZeroReferenceLoss

def train_v1():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlatZero3DCE().to(device)
    criterion = ZeroReferenceLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print("Starting v1 training (Baseline Flat Architecture)...")
    
    # This is a placeholder for the actual data loading logic
    # In a real scenario, you'd import your dataloaders from src.data
    
    model.train()
    # for epoch in range(100):
    #     for low_light_data in dataloader:
    #         optimizer.zero_grad()
    #         A, enhanced = model(low_light_data)
    #         loss = criterion(A, enhanced, low_light_data)
    #         loss.backward()
    #         optimizer.step()

if __name__ == "__main__":
    train_v1()
