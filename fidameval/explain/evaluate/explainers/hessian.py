import torch


def hessian(model, input_embs, _baseline):
    def model_forward(inputs_embeds):
        return model(inputs_embeds=inputs_embeds)

    model_hessian = torch.autograd.functional.hessian(model_forward, input_embs[0])

    # Sum over embedding dimension
    return model_hessian.sum([1, 3]).detach()


def hessian_input(model, input_embs, _baseline):
    def model_forward(inputs_embeds):
        return model(inputs_embeds=inputs_embeds)

    model_hessian = torch.autograd.functional.hessian(model_forward, input_embs[0])

    model_hessian_input = model_hessian * input_embs

    # Sum over embedding dimension
    return model_hessian_input.sum([1, 3]).detach()
