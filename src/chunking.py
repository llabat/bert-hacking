import pandas as pd 
from transformers import AutoTokenizer

def chunk_text(row : dict, tokenizer, max_length_capped : int, overlap: int, col_text : str, col_id : str) -> dict:
    row = row.copy()
    list_of_words = row[col_text].split(" ")
    list_of_words_tokenized = [
        [
            t for t in tokenizer(w).input_ids 
            if t not in [tokenizer.cls_token_id, tokenizer.sep_token_id]
        ]
        for w in list_of_words
    ]
    other_meta_data = {k:v for k,v in row.items() if k not in [col_text, col_id] }
    output  = []

    current_chunk = ""
    len_current_chunk_tokenized = 0
    i = 0

    while i < len(list_of_words): 
        word = list_of_words[i]
        len_tokenized_word = len(list_of_words_tokenized[i])

        if len_current_chunk_tokenized + len_tokenized_word < max_length_capped:
            current_chunk += word + " "
            len_current_chunk_tokenized += len_tokenized_word
            i+=1
        else: 
            output += [{
                col_text : current_chunk[:-1], # remove last space 
                col_id : row[col_id],
                f"{col_id}-chunk": f'{row[col_id]}-{len(output)}',
                **other_meta_data
            }] 
            current_chunk = ""
            len_current_chunk_tokenized = 0
            i -= overlap
    return pd.DataFrame(output)

def chunk_df(df : pd.DataFrame, tokenizer, max_length_capped : int, overlap: int, col_text: str, col_id: str) -> pd.DataFrame:
    output = (
        df.reset_index()
        .apply(lambda r : chunk_text(r, tokenizer, max_length_capped, overlap, col_text, col_id), axis = 1) 
    )
    return pd.concat(output.to_list()).set_index(col_id)

data_folder = "./data"
data_file = "ideology_news-dataset_for_training.csv"
parameters = {
    'max_length_capped' : 400,
    'overlap' : 50,
    'col_text': "content", 
    'col_id': "ID"
}
output_file = f"{data_file.removesuffix('.csv')}_{parameters['max_length_capped']}_{parameters['overlap']}.csv"

df_to_chunk = pd.read_csv(f'{data_folder}/{data_file}')
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

(
    chunk_df(df_to_chunk, tokenizer, **parameters)
    .to_csv(f'{data_folder}/{output_file}')
)