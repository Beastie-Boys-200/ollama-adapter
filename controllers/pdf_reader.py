from pypdf import PdfReader
from pathlib import Path



def read_pdf(pdf_path: Path) -> list[str]:
    reader = PdfReader(pdf_path) 

    all_text: str = ''

    for page in reader.pages:
        for row in page.extract_text(exctraction_mode='layout').split("\n"):
            if len(row) < 3:
                continue
            all_text += row
            if row[-1] == '-':
                all_text += ''
            else: 
                all_text += ' '

    split_sent: list[str] = []
    for sent in all_text.split('.'):
        if not len(sent.strip()):
            continue
        split_sent.append(sent.strip())
    
    result: list[str] = [ ". ".join(split_sent[_:_+30]) for _ in range(0, len(split_sent), 30) ]
        

    return result 
    





if __name__ == "__main__":

    text = read_pdf(Path('./pdf_samples/example1.pdf'))
    print(text)
    print(len(text))
