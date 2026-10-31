# StyleGAN-NADA Reimplementation

Реимплементация метода редактирования изображений с помощью текста из статьи
**StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators**.

## Статус

Проект в разработке. Реализован полный цикл обучения, идёт работа над экспериментами.

## Что реализовано

- Inference-пайплайн и цикл обучения StyleGAN2 с направленным CLIP-лоссом
- Ансамбль CLIP-моделей (ViT-B/16 + ViT-B/32), мультипромпты
- Три метода регуляризации: L2, Patch-based loss, LPIPS
- Адаптивная разморозка слоёв на основе оптимизации в W+
- Система экспериментов (ablation study) с отключением компонентов
- Анализ градиентов по слоям, heatmap обучения блоков, diversity

## Установка

### 1. Клонирование репозитория

Проект использует [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) как git submodule.

```bash
git clone --recursive https://github.com/KBaron34/StyleGAN-NADA-reimplementation.git
cd StyleGAN-NADA-reimplementation
```

Если репозиторий уже склонирован без `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Conda-окружение

```bash
conda env create -f environment.yml
conda activate stylegan-nada-reg
```

### 3. Предобученная модель

```bash
mkdir -p StyleGAN2
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -O StyleGAN2/ffhq.pkl
```

## Запуск

### Одиночный эксперимент (baseline)

```bash
python scripts/run_single.py
```

### Все эксперименты (ablation study)

```bash
python scripts/run_experiments.py
```

Результаты сохраняются в `results/`: метрики (`metrics.csv`), графики, сгенерированные изображения, checkpoint'ы.

## Структура проекта

```
configs/          # TrainConfig, Experiment, список экспериментов
src/
  utils/          # текст, изображения, freeze, checkpoint
  models/         # загрузка StyleGAN2 и CLIP
  losses/         # directional CLIP loss, регуляризации
  training/       # слои, оптимизаторы, trainer
  analysis/       # метрики, графики
scripts/          # run_single.py, run_experiments.py
stylegan2-ada-pytorch/   # submodule (NVIDIA)
```

## Ссылки

- [Оригинальная статья StyleGAN-NADA](https://arxiv.org/pdf/2108.00946)
- [Литературный обзор на Habr](https://habr.com/ru/articles/943034/)

## Стек

PyTorch, StyleGAN2-ADA, CLIP, LPIPS