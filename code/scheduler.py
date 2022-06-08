import torch.optim.lr_scheduler as lr_scheduler
import math

def create_scheduler(args, optimizer):
    """
    define scheduler
    """
    def poly_schd(epoch):
        return math.pow(1 - epoch / args.num_epoch, args.poly_exp)

    if args.scheduler == "lambda":
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd)
    elif args.scheduler == "cosineanneal":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.T_max, eta_min=args.eta_min
        )
    elif args.scheduler == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30])
    elif args.scheduler == "cosinerestart":
         scheduler = lr_scheduler.CosineAnnealingWarmUpRestarts(
            optimizer, T_0=1600, T_mult=1, eta_max=0.002, T_up =800, gamma = 0.5)

    return scheduler