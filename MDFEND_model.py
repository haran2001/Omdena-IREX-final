import torch
from transformers import BertModel
from faknow.data.process.text_process import TokenizerFromPreTrained
from faknow.model.content_based.mdfend import MDFEND
import warnings
import os
import gdown

class NewsClassifier:
    def __init__(self):
        self.max_len = 250
        self.bert = 'dccuchile/bert-base-spanish-wwm-cased'
        self.tokenizer = TokenizerFromPreTrained(self.max_len, self.bert)
        self.domain_num = 11

        self.MODEL_SAVE_PATH = self.download_model()
        self.MDFEND_MODEL = self.load_model()

    def download_model(self):
        script_dir = os.path.dirname(__file__)  # the cwd relative path of the script file
        rel_path = 'models/model_10_experts_20_epoch_best.pth'
        rel_to_cwd_path = os.path.join(script_dir, rel_path)

        if not os.path.exists(rel_to_cwd_path):
            # Download the model from Google Drive
            # https://drive.google.com/file/d/17u8fXwxm5JVWqEJwdcxzea2LhVl0KR5m/view?usp=sharing
            url = "17u8fXwxm5JVWqEJwdcxzea2LhVl0KR5m"
            gdown.download(id=url, quiet=True)

        return rel_to_cwd_path

    def load_model(self):
        model = MDFEND(self.bert, self.domain_num)
        model.load_state_dict(torch.load(f=self.MODEL_SAVE_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model

    def predict(self, text, domain=None):
        if domain is None:
            warnings.warn('The news domain was not identified. The model accuracy has been reduced.')
            domain = 0
        inputs = self.tokenizer(text)

        with torch.no_grad():
            outputs = self.MDFEND_MODEL(inputs['token_id'], inputs['mask'], torch.tensor(domain))

        return outputs.item()*100
