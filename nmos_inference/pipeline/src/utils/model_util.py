import torch
import transformers


# could be substituted by calculating in the forward function
def diy_loss(pred, label):
    loss = torch.nn.CrossEntropyLoss()
    loss = sum([loss(pred[:, i, :].squeeze(), label[:, i]) for i in range(pred.shape[1])])
    return loss


def diy_optimizer(model, lr, weight_decay, epsilon, momentum, correct_bias):
    no_decay = ["bias", "LayerNorm.weight"]
    opt_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.AdamW(opt_params, lr=lr, eps=epsilon,
                                   correct_bias=correct_bias)
    return optimizer
