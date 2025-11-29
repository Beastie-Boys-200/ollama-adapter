from .llm_planer import validate_with_metadata
from .llm_recruter import llm_recruter








def main() -> None:
    print("=== Metadata-aware validators (Ollama + Pydantic) ===")
    print("Input: prompt + flags has_image / has_doc.")
    print("If prompt empty but file attached -> auto prompt and no validation.")
    print("Otherwise -> meaningful + routing validators.\n")

    while True:
        prompt = input("User prompt (empty = exit): ").strip()

        img_flag = input("Image attached? [y/N]: ").strip().lower().startswith("y")
        doc_flag = input("Document attached? [y/N]: ").strip().lower().startswith("y")

        result = validate_with_metadata(prompt, has_image=img_flag, has_doc=doc_flag)

        print("\n--- Result ---")
        print(f"normalized_prompt: {result['normalized_prompt']!r}")
        print(f"auto_prompt: {result['auto_prompt']!r}\n")

        print("[meaningful]")
        print(result["meaningful"])

        print("\n[routing]")
        print(result["routing"])
        print("-----------------------------\n")

        if result["routing"] is None:
            print(result['meaningful'].text)
            continue
        if not result['routing']:
            print(result['routing'].text)
            continue


        

        paths = llm_recruter(query=prompt)
        print(paths.output)
        print("Process paths")
        print()
        print()
        print()
        print()

        






        


if __name__ == "__main__":
    main()

