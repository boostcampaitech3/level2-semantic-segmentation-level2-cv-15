import math

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_opt_sche(args, model):
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

    def poly_schd(epoch):
        return math.pow(1 - epoch / args.epochs, args.poly_exp)

    # if args.scheduler == "lambda":
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd)
    # elif args.scheduler == "cosineanneal":
    #     scheduler = lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=args.T_max, eta_min=args.eta_min
    #     )
    # elif args.scheduler == "step":
    #     scheduler = lr_scheduler.StepLR(
    #         optimizer, step_size=args.step_size, gamma=args.gamma
    #     )
    # elif args.scheduler == "multistep":
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30])

    return optimizer#, scheduler