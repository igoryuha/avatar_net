import torch


def moments(tensor, eps=1e-5):
    mean = tensor.mean((2,3), keepdim=True)
    var = tensor.var((2,3), keepdim=True, unbiased=False) + eps
    return mean, var


def AdaIN(content, style, eps=1e-05):
    """
    aligns the mean and variance of the content
    feature maps to those of the style feature maps
    """
    c_mean, c_var = moments(content)
    s_mean, s_var = moments(style)
    c_normalized = (content - c_mean) / torch.sqrt(c_var)
    return torch.sqrt(s_var) * c_normalized + s_mean