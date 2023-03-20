# mecab_huggingface_tokenizer
## Intro
띄어쓰기를 기준으로 tokenizing하는 huggingface의 tokenizer들과 mecab을 활용한 tokenizer 클래스입니다.

## Caution
* 해당 토크나이징 방식을 활용 했을 경우 pretrain 모델에 따라 finetune 성능이 떨어질 수도 있습니다.

## How to use
```python
from .src.mecab_huggingface_tokenizer import MecabHuggingfaceTokenizerFast

pretrain_path = "klue/roberta-base"

custom_tokenizer = MecabHuggingfaceTokenizerFast(
    pretrained_path=pretrain_path,
    max_length=256,
    device="cuda",
)

target_text = "대우조선해양이 오는 3월 사업을 확장한다"

mecab_tokenizer_result = custom_trainer(target_text)
```

## Compare mecab_huggingface_tokenizer to original huggingface tokenizer
```python
from transformers import AutoTokenizer
from .src.mecab_huggingface_tokenizer import MecabHuggingfaceTokenizerFast

pretrain_path = "klue/roberta-base"

tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
custom_tokenizer = MecabHuggingfaceTokenizerFast(
    pretrained_path=pretrain_path,
    max_length=256,
    device="cuda",
)

target_text = "대우조선해양이 오는 3월 사업을 확장한다"

huggingface_tok_result = tokenizer(target_text)
mecab_tokenizer_result = custom_tokenizer(input_sentence=target_text)

# huggingface tokenizer를 통과한 결과 토큰들
print(tokenizer.convert_ids_to_tokens(huggingface_tok_result["input_ids"]))
# mecab huggingface tokenizer를 통과한 결과 토큰들
print(tokenizer.convert_ids_to_tokens(mecab_tokenizer_result["input_ids"]))
```

## Contact
* jminju1111@gmail.com
