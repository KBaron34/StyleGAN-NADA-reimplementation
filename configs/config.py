from dataclasses import dataclass


@dataclass
class TrainConfig:
    """
    Конфигурация обучения генератора StyleGAN2 с CLIP-адаптацией.

    :param model_names: список имен моделей CLIP
    :param text_source: исходный текстовый промпт
    :param text_target: целевой текстовый промпт
    :param epochs: количество эпох обучения генератора
    :param lr: скорость обучения для оптимизатора параметров генератора
    :param lr_w: скорость обучения для оптимизатора векторов из пространства W+
    :param gamma: коэффициент затухания скорости оптимизации параметров генератора
    :param net: CNN-модель, на признаках которой считается LPIPS
    :param lambda_patch: коэффициент патч-регуляризации
    :param lambda_l2: коэффициент l2-регуляризации
    :param lambda_lpips: коэффициент LPIPS-регуляризации
    :param max_norm: максимальная норма градиентов
    :param weights: вклад модели CLIP в итоговый эмбеддинг
    :param experiment_name: имя эксперимента
    :param save_dir: директория для сохранения результатов
    :param load_dir: директория для загрузки checkpoint
    :param k: количество слоев которые обучаются
    :param patch_size: размер квадрата патча
    :param num_patches: количество патчей
    :param batch_size: размерность батча для обучения
    :param sample_size: размерность генерируемого сэмпла для визуализации
    :param div_size: размерность генерируемого сэмпла для вычисления diversity
    :param batch_size_w: размерность батча для оптимизации векторов из пространства W+
    :param flag_clip: флаг использования 1 (False) или 2 (True) моделей CLIP
    :param flag_prompt: флаг использования 1 (False) или нескольких (True) промптов
    :param flag_reg_l2: флаг использования l2-регуляризации
    :param flag_reg_patch: флаг использования patch-лосса
    :param flag_reg_lpips: флаг использования lpips-лосса
    :param flag_adaptive_layers: флаг использования адаптивного выбора обучаемых слоев
    :param flag_gradient_clipping: флаг использования ограничения нормы градиента
    :param RAND: фиксирует генератор случайных чисел
    """

    model_names: list[str] = None
    text_source: str = "human"
    text_target: str = "werewolf"
    epochs: int = 301
    lr: float = 0.003
    lr_w: float = 0.02
    gamma: float = 0.98
    net: str = "alex"
    lambda_patch: float = 0.15
    lambda_l2: float = 0.001
    lambda_lpips: float = 0.01
    max_norm: float = 0.7
    weights: list[float] = None
    experiment_name: str = "baseline"
    save_dir: str = "results"
    load_dir: str = None
    stylegan_ckpt: str = None
    k: int = 12
    patch_size: int = 32
    num_patches: int = 8
    batch_size: int = 2
    sample_size: int = 4
    div_size: int = 20
    batch_size_w: int = 8
    flag_clip: bool = True
    flag_prompt: bool = True
    flag_reg_l2: bool = True
    flag_reg_patch: bool = True
    flag_reg_lpips: bool = True
    flag_adaptive_layers: bool = True
    flag_gradient_clipping: bool = True
    RAND: int = 2

    def __post_init__(self):
        """
        Устанавливает значения по умолчанию для списков.
        """
        if self.stylegan_ckpt is None:
            self.stylegan_ckpt = "StyleGAN2/ffhq.pkl"
        if self.model_names is None:
            self.model_names = ["ViT-B/16", "ViT-B/32"]
        if self.weights is None:
            self.weights = [1.0, 1.0]
