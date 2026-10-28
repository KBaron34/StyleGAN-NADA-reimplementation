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