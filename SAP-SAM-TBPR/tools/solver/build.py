import torch

from .lr_scheduler import LRSchedulerWithWarmup


def make_optimizer(args, model):
    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.base_lr
        weight_decay = args.weight_decay
        if "bias" in key:
            lr = args.base_lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.base_lr, momentum=args.sgd_momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.base_lr,
            betas=(args.adam_alpha, args.adam_beta),
            eps=1e-8,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.base_lr,
            betas=(args.adam_alpha, args.adam_beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def make_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.steps,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epochs,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
