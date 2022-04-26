import torch.optim as optim

def create_optimizer(args, model):
    """
    define optimizer & scheduler
    """
    param_groups = model.parameters()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=False,
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Not a valid optimizer")

    return optimizer