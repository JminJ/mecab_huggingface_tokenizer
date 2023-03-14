from typing import List, Tuple

from transformers import AutoTokenizer
from mecab import MeCab


class MecabHuggingfaceTokenizerFast:
    def __init__(self, pretrained_path: str, max_length: int, device: str):
        self.huggingface_tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.mecab = MeCab()
        self.max_length = max_length
        self.device = device

    def __call__(self, input_sentence):
        # mecab을 통해 target 문장을 1차 처리
        if type(input_sentence) == str:
            input_sentence = [input_sentence]
            tok_result = self._tokenize_one(input_sentence)

            return tok_result

        elif type(input_sentence) == list:
            tok_result = self._tokenize_many(input_sentence)

            return tok_result

        else:
            raise TypeError("Please, set your 'input_sentence' value type str or list.")

    def _tokenize_one(self, input_sentence: str):
        mecab_offset_result, mecab_surface_result = self.__mecab_oper_preprocessing(
            input_texts=input_sentence
        )

        # mecab을 통해 나눠진 문장을 " " 기준으로 join
        mecab_texts = [" ".join(s) for s in mecab_surface_result]

        # mecab으로 1차 처리된 문장을 original huggingface tokenizer를 통해 토크나이징
        tok_result = self.huggingface_tokenizer(
            mecab_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        tok_result["input_ids"] = tok_result["input_ids"][0]

        return tok_result

    def _tokenize_many(self, input_sentences: List):
        mecab_offset_result, mecab_surface_result = self.__mecab_oper_preprocessing(
            input_texts=input_sentences
        )

        # mecab을 통해 나눠진 문장을 " " 기준으로 join
        mecab_texts = [" ".join(s) for s in mecab_surface_result]

        # mecab으로 1차 처리된 문장을 original huggingface tokenizer를 통해 토크나이징
        tok_result = self.huggingface_tokenizer(
            mecab_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        return tok_result

    def __mecab_oper_preprocessing(self, input_texts: List) -> Tuple[List, List]:
        mecab_result = [self.mecab.parse(t) for t in input_texts]

        # offset, surface 처리
        mecab_offset_result = []
        mecab_surface_result = []
        for i in range(len(mecab_result)):
            temp_result = mecab_result[i]
            temp_offset_result = []
            temp_surface_result = []
            for t in temp_result:
                temp_offset = [t[0][0], t[0][1]]
                temp_offset_result.append(temp_offset)

                temp_surface_result.append(t[1])

            mecab_offset_result.append(temp_offset_result)
            mecab_surface_result.append(temp_surface_result)

        return mecab_offset_result, mecab_surface_result


if __name__ == "__main__":
    # pretrain_path = "monologg/koelectra-base-v3-discriminator"
    pretrain_path = "klue/roberta-base"

    custom_tokenizer = MecabHuggingfaceTokenizerFast(
        pretrained_path=pretrain_path,
        max_length=256,
        device="cuda",
    )

    target_text = "대우조선해양이 오는 3월 사업을 확장한다"

    mecab_tokenizer_result = custom_tokenizer(input_sentence=target_text)

    huggingface_tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    tokenizer_result = huggingface_tokenizer(target_text)
    print(mecab_tokenizer_result["input_ids"])

    print(huggingface_tokenizer.convert_ids_to_tokens(mecab_tokenizer_result["input_ids"]))
    print(huggingface_tokenizer.convert_ids_to_tokens(tokenizer_result["input_ids"]))
