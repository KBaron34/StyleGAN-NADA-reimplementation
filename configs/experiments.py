from dataclasses import dataclass


@dataclass
class Experiment:
    """
    Конфигурация одного эксперимента.

    :param name: имя эксперимента
    :param flag_clip: флаг использования ансамбля CLIP-моделей (2 модели) или одной модели
    :param flag_prompt: флаг использования нескольких промптов или одного
    :param flag_reg_l2: флаг использования L2-регуляризации
    :param flag_reg_patch: флаг использования patch-регуляризации
    :param flag_reg_lpips: флаг использования LPIPS-регуляризации
    :param flag_adaptive_layers: флаг использования адаптивного выбора обучаемых слоев (W+)
    :param flag_gradient_clipping: флаг использования ограничения нормы градиента
    """

    name: str
    flag_clip: bool = True
    flag_prompt: bool = True
    flag_reg_l2: bool = True
    flag_reg_patch: bool = True
    flag_reg_lpips: bool = True
    flag_adaptive_layers: bool = True
    flag_gradient_clipping: bool = True


def get_experiments() -> list:
    """
    Возвращает список конфигураций для одного эксперемента.

    :return: список объектов Experiment (baseline + 7 конфигураций с отключенными компонентами)
    """
    return [
        Experiment(name="baseline"),
        Experiment(name="no_clip", flag_clip=False),
        Experiment(name="no_prompt", flag_prompt=False),
        Experiment(name="no_l2", flag_reg_l2=False),
        Experiment(name="no_patch", flag_reg_patch=False),
        Experiment(name="no_lpips", flag_reg_lpips=False),
        Experiment(name="no_adaptive", flag_adaptive_layers=False),
        Experiment(name="no_grad_clip", flag_gradient_clipping=False),
    ]
