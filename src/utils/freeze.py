def freeze_module(module) -> None:
    """
    Замораживает все параметры переданного модуля, отключая вычисление градиентов.

    :param module: модуль, параметры которого необходимо заморозить

    :return: None
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module) -> None:
    """
    Размораживает все параметры переданного модуля, включая вычисление градиентов.

    :param module: модуль, параметры которого необходимо разморозить

    :return: None
    """
    for param in module.parameters():
        param.requires_grad = True
