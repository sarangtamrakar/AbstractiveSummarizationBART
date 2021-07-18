import torch.cuda
from transformers import BartForConditionalGeneration, BartTokenizer
from Config import ConfigClass
from log import LogClass


class PredictionClass:
    def __init__(self):
        # Getting config data
        self.ConfigObj = ConfigClass("params.yaml")
        self.ConfigData = self.ConfigObj.Loading_Config()
        self.logFileName = self.ConfigData['LoggingFileName']
        self.ModelName = self.ConfigData['Loading']['Model_name']
        self.TokenizerName = self.ConfigData['Loading']['TokenizerDir']
        self.context_max_len = self.ConfigData['Generate']['context_max_len']
        self.summary_max_len = self.ConfigData['Generate']['summary_max_len']
        self.num_beams = self.ConfigData['Generate']['num_beams']
        self.repetition_penalty = self.ConfigData['Generate']['repetition_penalty']
        self.length_penalty = self.ConfigData['Generate']['length_penalty']
        self.early_stopping = self.ConfigData['Generate']['early_stopping']

        self.LoggerObj = LogClass(self.logFileName)
        self.LoggerObj.Logger("Loading Tokenizer & Model")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BartTokenizer.from_pretrained(self.TokenizerName)
        self.Model = torch.load(self.ModelName)


    def CleanData(self, text):
        try:
            self.LoggerObj.Logger("Cleaning the Data")
            new_text = " ".join(text.split(". "))
            return new_text
        except Exception as e:
            self.LoggerObj.Logger("Exception Occured in CleanData method of PredictionClass : "+str(e))
            return "Exception Occured in CleanData method of PredictionClass : "+str(e)

    def Prediction(self, text):
        try:
            global encodings
            # clean the text
            # ctext = self.CleanData(text)
            input_ids = []
            attention_mask = []

            encodings = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=self.context_max_len,
                is_split_into_words=False,
                return_tensors="pt",
                return_attention_mask=True,
            )

            input_ids.append(encodings["input_ids"].to(self.device))
            attention_mask.append(encodings["attention_mask"].to(self.device))

            # concatenate the tensors
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)

            # merge data into dictionary format
            dic = {"input_ids": input_ids, "attention_mask": attention_mask}

            # passing data to Model
            res = self.Model.generate(
                **dic,
                max_length=self.summary_max_len,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                early_stopping=self.early_stopping
            )

            # decode the Prediction...
            preds = [
                self.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in res
            ]

            # binding into dictionary..
            final_result = {"prediction": preds, "Context": str(text)}
            self.LoggerObj.Logger("Get the final Prediction : "+str(final_result))
            return final_result
        except Exception as e:
            self.LoggerObj.Logger("Exception occured : " + str(e))
            return "Exception occured : " + str(e)
