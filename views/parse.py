from ddgs import DDGS


def get_search_links(query, max_results=10, verbose=False):
    """
    Функция для получения ссылок из поиска DuckDuckGo

    :param query: Поисковый запрос
    :param max_results: Максимальное количество результатов (по умолчанию 10)
    :param verbose: Выводить ли процесс поиска (по умолчанию False)
    :return: Список ссылок (list)
    """
    links = []

    try:
        if verbose:
            print(f"Ищу: {query}")
            print("Подождите, идет поиск...")

        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)

        for result in results:
            link = result["href"]
            links.append(link)

            if verbose:
                title = result["title"]
                print(f"Найдено: {title}")
                print(f"  URL: {link}\n")

        if verbose:
            print(f"{'='*80}")
            print(f"Всего найдено {len(links)} ссылок")

    except Exception as e:
        if verbose:
            print(f"\nОшибка: {e}")
            import traceback

            traceback.print_exc()

    return links


# Пример использования
if __name__ == "__main__":
    # Вариант 1: Простой вызов
    links = get_search_links("python tutorial", max_results=10, verbose=True)
    print(links)

    # Вариант 2: Без вывода в консоль
    # links = get_search_links("python tutorial", max_results=5, verbose=False)
    # print(links)  # ['url1', 'url2', 'url3', ...]
