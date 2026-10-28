import re
from src.utils.templates import TEMPLATES


def cleaned_text(text) -> str:
    """
    Чистит текст.

    :param text: входной текст (может быть None, числом, строкой)

    :return: очищенная строка в нижнем регистре
    """
    text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^а-яёa-z0-9\s.,*!?:-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[^\w]+", "", text)
    text = text.strip(" .")
    return text


def make_prompt_list(
    word: str, templates: list[str] = TEMPLATES, flag_prompt: bool = True
) -> list[str]:
    """
    Создает список промптов.

    :param word: слово для добавления в промпт
    :param templates: список шаблонов промптов
    :param flag_prompt: флаг использования 1 (False) или нескольких (True) промптов

    :return: список промптов
    """
    word = cleaned_text(text=word)

    if flag_prompt:
        return [t.format(word) for t in templates]
    return [word]
