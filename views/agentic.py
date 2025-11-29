from .llm_planer import validate_with_metadata
from .llm_router import llm_router  # новий третій агент з route: int

TEST_CASES = [
    # 1. Базовий shallow (0)
    ("What is Docker?", False, False, 0),
    # 2. Простий концептуальний запит (0)
    ("Explain the difference between Docker and virtual machines.", False, False, 0),
    # 3. Явний web / deep research (1)
    ("Do deep research on Docker architecture using web sources.", False, False, 1),
    # 4. Неявний web через "latest/current" (1)
    ("What are the latest Docker trends in 2025?", False, False, 1),
    # 5. Ще один web-кейс: поточні дані (1)
    ("Check the current Bitcoin price today.", False, False, 1),
    # 6. Docs-intent БЕЗ прикріплених файлів (має піти в 2 через router)
    ("What is Docker based on my docs?", False, False, 2),
    # 7. Docs-intent БЕЗ файлів, але іншою формою (2)
    ("Summarize the architecture according to my PDF documentation.", False, False, 2),
    # 8. Image-intent БЕЗ прикріплених файлів (має піти в 3 через router)
    ("Based on my screenshots, is this Docker setup correct?", False, False, 3),
    # 9. Image-intent іншою формою (3)
    ("In the photo I sent before, what is on the left side?", False, False, 3),
    # 10. Docs pipeline з реальним доком (has_doc=True, route=2, router не викликається)
    ("Summarize this document.", False, True, 2),
    # 11. Docs pipeline: текст загальний, але є док (можеш перевірити, що все одно route=2)
    ("Explain Docker.", False, True, 2),
    # 12. Images pipeline з реальним зображенням (has_image=True, route=3, router не викликається)
    ("Describe what is shown in this picture.", True, False, 3),
    # 13. Images pipeline: загальний текст, але є картинка
    ("Is this the official Docker logo?", True, False, 3),
    # 14. Авто-промпт: порожній текст + тільки зображення (normalized_prompt -> 'Describe the attached image.')
    ("", True, False, 3),
    # 15. Авто-промпт: порожній текст + тільки документ (normalized_prompt -> 'Summarize the attached document.')
    ("", False, True, 2),
    # 16. Повна порожнеча: ні тексту, ні файлів (має завалитися на meaningful)
    ("", False, False, -1),
    # 17. Нісенітниця / keyboard mashing (має завалитися на meaningful)
    ("asdqwe!!!@@@", False, False, -1),
    # 18. Надто розмите "help" (пройде meaningful, але завалиться routing)
    ("Help me", False, False, -1),
    # 19. Розмите "Fix this" без контексту (routing має сказати state=false)
    ("Fix this", False, False, -1),
    # 20. Контекстно коректний image-кейс з метаданими (перевіряє узгодженість з валідаторами)
    ("Explain this picture in detail.", True, False, 3),
]


def main() -> None:
    print("=== Metadata-aware validators (Ollama + Pydantic) ===")
    print("Input: prompt + flags has_image / has_doc.")
    print("If prompt empty but file attached -> auto prompt and no validation.")
    print("Otherwise -> meaningful + routing validators.\n")

    for i, (prompt, img_flag, doc_flag, correct) in enumerate(TEST_CASES, start=1):
        print(f"===== TEST {i} =====")
        print(prompt)
        print("Is img:", img_flag)
        print("Is doc:", doc_flag)
        print()

        # prompt = input("User prompt (empty = exit): ").strip()
        # if prompt == "":
        #     break

        # img_flag = input("Image attached? [y/N]: ").strip().lower().startswith("y")
        # doc_flag = input("Document attached? [y/N]: ").strip().lower().startswith("y")

        result = validate_with_metadata(prompt, has_image=img_flag, has_doc=doc_flag)

        normalized_prompt = result["normalized_prompt"]
        auto_prompt = result["auto_prompt"]
        meaningful = result["meaningful"]  # Validation
        routing_validation = result["routing"]  # Validation | None

        print("--- Result ---")
        print(f"normalized_prompt: {normalized_prompt!r}")
        print(f"auto_prompt: {auto_prompt!r}\n")

        print("[meaningful]")
        print(meaningful)

        print("\n[routing]")
        print(routing_validation)

        # 1) Якщо перший валідатор завалився – показуємо текст і не йдемо далі
        if not meaningful.state:
            print(f'[Agent #1 failed]       {"PASS" if correct == -1 else "ERROR"}')
            print(meaningful.text)
            continue

        # 2) Якщо другий валідатор не спрацьовував (None) або завалився
        if routing_validation is None:
            print(
                f'[Agent #2 did not run]       {"PASS" if correct == -1 else "ERROR"}'
            )
            # за твоєю логікою тут можна показати текст meaningful/text, але він уже ок
            continue

        if not routing_validation.state:
            print("[Agent #2 failed]")
            print(routing_validation.text)
            continue

        route = -1
        if img_flag:
            route = 3
        elif doc_flag:
            route = 2
        else:
            route = int(llm_router(prompt).output.route)

        print("\n=== Route:   ", end="")
        print(route, end="")
        print(f'    {"PASS" if correct == route else "ERROR"}')
        print("\n-----------------------------")
        print("\n\n\n")


if __name__ == "__main__":
    main()
