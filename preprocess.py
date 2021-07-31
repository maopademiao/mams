import yaml
from data_process.data_process import data_process

config = yaml.safe_load(open('config.yml'))

data_process(config)

# train: 11186
# dev: 2668
# word: 7899
# sentence_max_len:70
# aspect_max_len:10
# context_max_len:70  context就是将aspect terms [id] 换成 [ASPECT_INDEX = 2]
# bert_max_len:106  [CLS] text [SEP] aspect terms [SEP]
# td_left_max_len:65  包含 aspect terms 的左半部分
# td_right_max_len:66  包含 aspect terms 的右半部分 - 转置
