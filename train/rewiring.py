# train/rewiring.py

def apply_rewiring(model, strategy):
    if hasattr(model, "rewire"):
        model.rewire(strategy)
    else:
        for module in model.modules():
            if hasattr(module, "rewire"):
                module.rewire(strategy)
