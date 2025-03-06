import parse_tree as pt
from transformers import BertTokenizer
import json
from vocab import Vocab


# tokens = tokenizer.tokenize(text)
def convert(text, words, tokenizer):
    tokens = text.split()

    aspects = []
    for word in words:
        # 找到单词的from和to
        start_index = None
        for i in range(len(tokens)):
            if tokens[i] == word.split()[0] and tokens[i:i + len(word.split())] == word.split():
                start_index = i
                break

        if start_index is not None:
            from_offset = start_index
            to_offset = start_index + len(word.split())  # -1

            # 输出结果
            # print(f"The word '{word}' starts at position {from_offset} and ends at position {to_offset}.")
            ans = {'term': word.split(), 'from': from_offset, 'to': to_offset, 'polarity': 'unknown'}
            aspects.append(ans)
        else:
            print(f"The word '{word}' was not found in the token sequence.")

    res = [{'token': tokens, 'aspects': aspects}]
    path = 'data/V2/MAMS'
    with open(path + '/test1.json', 'w') as f:
        json.dump(res, f, indent=4)

    pt.preprocess_file(path + '/test1.json')

# vo = Vocab.load_vocab('data/V2/MAMS/vocab_pol.vocab')
# print(vo.itos)
# ['negative', 'neutral', 'positive']
