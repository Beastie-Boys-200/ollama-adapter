import markdown
from weasyprint import HTML, CSS

class MkPDF:

    @staticmethod
    def md_to_pdf(md_text, output_path):

        html = markdown.markdown(md_text, extensions=['fenced_code','tables'])

        css = """
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            white-space: pre-wrap; /* перенос длинных строк */
            word-wrap: break-word;
        }
        code {
            font-family: monospace;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 5px;
            text-align: center;
            word-wrap: break-word;
        }
        """

        HTML(string=html).write_pdf(output_path, stylesheets=[CSS(string=css)])


MkPDF().md_to_pdf("""

# Тестовый документ Markdown → PDF

Это пример документа, который включает различные элементы Markdown.

---

## 1. Обычный текст

Привет! Это абзац текста для проверки, как обычный текст отображается в PDF.

Второй абзац для проверки переноса строк и форматирования.

---

## 2. Списки

### Маркированный список:
- Пункт 1
- Пункт 2
- Пункт 3

### Нумерованный список:
1. Первый
2. Второй
3. Третий

---

## 3. Код

### Однострочный код:
Используем `print("Hello World")` прямо в тексте.

### Блок кода:

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))



# Пример таблицы

| Имя      | Возраст | Город           |
|----------|:-------:|----------------|
| Иван     |   28    | Москва         |
| Ольга    |   31    | Санкт-Петербург|
| Михаил   |   22    | Казань         |
""", "output.pdf")

