import torch.optim as optim
from opacus import PrivacyEngine

def wrap_with_dp(model, trainloader):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    privacy_engine = PrivacyEngine()


    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=1,
        target_epsilon=8,
        target_delta=1e-5,
        max_grad_norm=1.0,
    )
    print_epsilon(privacy_engine)

    return model, optimizer, privacy_engine

def print_epsilon(privacy_engine, delta=1e-5):
    """Track epsilon value during training."""
    try:
        epsilon = privacy_engine.get_epsilon(delta)
        print(f"Current epsilon: {epsilon:.2f}, delta: {delta:.5f}")
    except ValueError as e:
        print(f"Error calculating epsilon: {e}")