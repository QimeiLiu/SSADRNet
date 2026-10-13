"""Factory and loss functions for the eight published Synapse ablations."""

import torch


def adamw_param_groups(model, weight_decay, no_decay_norm_bias=False):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if no_decay_norm_bias and (
                parameter.ndim <= 1 or "norm" in name or "bias" in name):
            no_decay.append(parameter)
        elif "gate" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_model(arch, n_class, img_size, model_scale="small"):
    """Build one of the eight archived models.

    Public C aliases intentionally hide historical experiment branch names while
    preserving the exact module topology required by the archived checkpoints.
    """
    arch = arch.lower()
    if arch == "base":
        from lib.networks import MaxViT_CASCADE
        cls, kwargs = MaxViT_CASCADE, {}
    elif arch == "base_msdec_sk":
        from lib.networks_mwsa import Single224_CASCADE_MSDec_SK
        cls, kwargs = Single224_CASCADE_MSDec_SK, {}
    elif arch == "datin_only":
        from lib.networks_mwsa import Single224_CASCADE_DATin_NoMS
        cls, kwargs = Single224_CASCADE_DATin_NoMS, {}
    elif arch == "sk_datin":
        from lib.networks_mwsa import Single224_CASCADE_SKDec_DATin
        cls, kwargs = Single224_CASCADE_SKDec_DATin, {}
    elif arch == "c_unified":
        from lib.networks_mwsa import Single224_CASCADE_JointChannelC23Weak
        cls, kwargs = Single224_CASCADE_JointChannelC23Weak, {}
    elif arch == "a_c_unified":
        from lib.networks_mwsa import Single224_CASCADE_SKDec_JointChannelC2
        cls, kwargs = Single224_CASCADE_SKDec_JointChannelC2, {"max_delta": 0.25}
    elif arch == "b_c_unified":
        from lib.networks_mwsa import Single224_CASCADE_DATin_ComplementaryC1
        cls, kwargs = Single224_CASCADE_DATin_ComplementaryC1, {}
    elif arch == "a_b_c_unified":
        from lib.networks_mwsa import Single224_CASCADE_SKDec_DATin_JointChannelC23
        cls, kwargs = Single224_CASCADE_SKDec_DATin_JointChannelC23, {}
    else:
        raise ValueError("unknown release architecture: %s" % arch)
    return cls(
        n_class=n_class, img_size=img_size, model_scale=model_scale, **kwargs)


def mutation_loss(
        predictions, label, ce_loss, dice_loss, subsets,
        lc1=0.3, lc2=0.7, ug_alpha=0.0, eps=1e-8):
    label_long = label.long()
    losses, certainty_logits = [], []
    for subset in subsets:
        if not subset:
            continue
        combined = sum((predictions[index] for index in subset), 0.0)
        losses.append(
            lc1 * ce_loss(combined, label_long)
            + lc2 * dice_loss(combined, label, softmax=True))
        if ug_alpha > 0:
            with torch.no_grad():
                probability = torch.softmax(combined, dim=1)
                entropy = -(
                    probability * torch.log(probability + eps)
                ).sum(dim=1).mean()
            certainty_logits.append(-ug_alpha * entropy)

    if ug_alpha > 0:
        weights = torch.softmax(
            torch.stack(certainty_logits), dim=0) * len(losses)
        return sum(weight * loss for weight, loss in zip(weights, losses))
    return sum(losses)
