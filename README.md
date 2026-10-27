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

## Ссылки

- [Оригинальная статья](https://arxiv.org/pdf/2108.00946)
- [Литературный обзор на Habr](https://habr.com/ru/articles/943034/)

## Стек

PyTorch, StyleGAN2-ADA, CLIP, LPIPS