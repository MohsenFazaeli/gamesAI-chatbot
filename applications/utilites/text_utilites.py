import string
# def remove_punctuation(text):
#     translator = str.maketrans('', '', string.punctuation+"؟")
#     text = text.translate(translator).replace('؟', '')
#     return text

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation.replace('؟', ''))
    return text.translate(translator).replace('؟', '').replace('،','')

if __name__ == '__main__':
    # Example usage:
    persian_text = "آیا این یک متن نمونه است؟ بله، این یک متن نمونه است!"
    text_without_punctuation_except_question_mark = remove_punctuation(persian_text)

    print("Original Persian Text:")
    print(persian_text)

    print("\nText without Punctuation (except ?):")
    print(text_without_punctuation_except_question_mark)
