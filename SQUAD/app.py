from flask import Flask,render_template
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField

app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

####################################################################
import torch
import torch.nn as nn

from transformers import T5Tokenizer
from torch import jit

tokenizer = T5Tokenizer.from_pretrained('t5-small')

init_token = tokenizer.pad_token
eos_token = tokenizer.eos_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

max_input_length = tokenizer.max_model_input_sizes['t5-small']

new_model = jit.load('t5_ts_qa_model.zip')

def translate_sentence2(sentence, eval_model, max_len = 50):
    
    eval_model.eval()
    eval_model = eval_model.float()

    src_indexes = [init_token_idx] + sentence + [eos_token_idx]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0)

    trg_indexes = [init_token_idx]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0)
        
        with torch.no_grad():
            
            output = eval_model(src_tensor, trg_tensor)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == eos_token_idx:
            break

    return trg_indexes[1:-1]

def return_answer(context,query):
    txt = 'context : ' + context.lower() + ' question : ' + query.lower()
    txt_tokens = tokenizer.tokenize(txt)
    txt_ids = tokenizer.convert_tokens_to_ids(txt_tokens)
    pred = translate_sentence2(txt_ids, new_model)
    pred_tokens = tokenizer.convert_ids_to_tokens(pred)
    
    return ''.join(pred_tokens)
####################################################################

class InfoForm(FlaskForm):
    
    context = StringField("Context")
    query = StringField("Query")
    
    submit = SubmitField("Submit")

@app.route('/',methods=['GET','POST'])
def index():

    result = False
    form = InfoForm()
    
    if form.validate_on_submit():

        result = return_answer(form.context.data,form.query.data)
        
        # form.context.data = ''
        # form.query.data = ''

    return render_template('basic.html',form=form,result=result)

if __name__ == '__main__':
    app.run(debug=True)