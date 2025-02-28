from transformers import AutoTokenizer
import json
import argparse

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import argparse
import json
from tqdm import tqdm
import warnings

from datasets.build import __factory

# CUDA_VISIBLE_DEVICES=7 python generate_cuhk_attributes.py --data_path ../sam-for-tbpr/ --chatglm_path ./chatglm2-6b/ --step 100000

class dataset_options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for Deep Cross Modal')
        self._par.add_argument('--dataset', type=str)
        self._par.add_argument('--wordtype', type=str)
        self._par.add_argument('--pkl_root', type=str)
        self._par.add_argument('--class_num', type=int)
        self._par.add_argument('--vocab_size', type=int)
        self._par.add_argument('--dataroot', type=str)
        self._par.add_argument('--mode', type=str)
        self._par.add_argument('--batch_size', type=int)
        self._par.add_argument('--caption_length_max', type=int)

class options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for datasets')
        self._par.add_argument('--step',default=1000, type=int)
        self._par.add_argument('--begin',default=0, type=int)
        self._par.add_argument('--quantize',default=16, type=int)
        self._par.add_argument('--dataset',default="CUHK-PEDES", type=str)
        self._par.add_argument('--data_path',default="./", type=str)
        self._par.add_argument('--chatglm_path',default="THUDM/chatglm2-6b", type=str)

if __name__ == "__main__":

    opt = options()._par.parse_args()

    print("Begin processing!")

    dataset = __factory[opt.dataset](root=opt.data_path)

    # setting prompt to chat with model
    prompt = '你好，我会给你一些英文句子，我希望你将英文句子切分成包含所有外貌描述的短句。' +\
        '并且我希望结果是python list 的形式,例如 ["black and grey jacket", "blue shirt", "black pants", "glasses" ]，' + \
        '请你记住这一点。下面是一个例子，the tall man is in business like attire. ' + \
        'he is wearing a blue collared shirt with long sleeves, dark pants and dark shoes. '+\
        'balding in the front, he is also wearing glasses. 我希望你直接输出结果：["black and grey jacket",'+\
        ' "blue shirt", "black pants", "glasses" ]'

    path = opt.chatglm_path

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # if upspeed or reduce GPU memory using quantize(8) or quantize(4)
    if opt.quantize == 16:
        model = AutoModel.from_pretrained(path, trust_remote_code=True, device='cuda').eval()
        #pass
    else:
        model = AutoModel.from_pretrained(path, trust_remote_code=True).quantize(opt.quantize).cuda().eval()
    
    history = [
        (
            prompt,
            '["black and grey jacket","blue shirt", "black pants", "glasses"]'
        ),
        (
            "this man is wearing a grey sweater, dark blue jeans, and shoes. he is also carrying a grey / black book bag on his back.",
            '["a grey sweater","dark blue jeans","shoes","a grey / black book bag"]'
        ),
        (
            "a man wearing a black jacket, black pants, red, black and white shoes and a gray shirt under his jacket.",
            '["a black jacket","black pants","red, black and white shoes","a gray shirt"]'
        ),
        (
            "The young woman, wearing a black cross-body bag, a brown down jacket, dark tights, and black leather shoes, walked by the roadside near bikes",
            '["a black cross-body bag","a brown down jacket","dark tights","black leather shoes"]'
        )
    ]

    # setting prompt to model
    response, history_ = model.chat(tokenizer, prompt, history=history)

    # give some examples to enhance the task prompt and save history
    text_description = "this man is wearing a grey sweater, dark blue jeans, and shoes. he is also carrying a grey / black book bag on his back."
    response, history_ = model.chat(tokenizer, text_description, history=history)
    #print(response)
    #print(history_)

    response, history_ = model.chat(tokenizer, "a man wearing a black jacket, black pants, red, black and white shoes and a gray shirt under his jacket.", history=history)
    #print(response)

    # process CUHK dataset
    attribute_list = []
    err_id_list = []
    end = min(opt.begin+opt.step,len(dataset.train))
    attribute_list_temp = []
    for idx in tqdm(range(opt.begin,end)):
        data = dataset.train[idx]
        cuhk_text = data[-1]
        key = data[-2][len(dataset.dataset_dir+"imgs/"):]
        try:
            warnings.filterwarnings("ignore")
            response, history_ = model.chat(tokenizer, cuhk_text, history=history)
            output = json.loads(response)
            attribute_list_temp.append(
                {"name":key,"attribute":output,"idx":idx,"sentence":cuhk_text}
            )
            #print(attribute_list_temp)
        except:
            print("error:",idx)
            err_id_list.append(idx)
            with open(opt.dataset+"_err_list.txt", 'a+') as f:
                try:
                    f.write("{}\t{}\t{}\n".format(idx,key,cuhk_text))
                except:
                    pass
            #print(e)
            #break
        
        finally:
            # save a checkpoint with step 2000
            if (idx+1) % 2000 == 0 or idx+1 == end:
                with open(opt.dataset+"_data_ckpt_{}.json".format(idx+1), 'w') as  f:
                    json.dump(attribute_list_temp, f)
                
                attribute_list += attribute_list_temp
                attribute_list_temp = []
            
            pass
    #     if idx > 50:
    #         break
    # save all the data
    with open(opt.dataset+"_data_final_{}.json".format(end), 'w') as  f:
        json.dump(attribute_list, f)
    
    print("End processing!")
