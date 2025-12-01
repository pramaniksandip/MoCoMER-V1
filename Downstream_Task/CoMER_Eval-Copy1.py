from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from torchvision.transforms import ToTensor
import torch
from PIL import Image
#from IPython.display import display
import pandas as pd
import os

#ckpt = '/workspace/nemo/data/RESEARCH/HMER_CoMER/CoMER/lightning_logs/version_9/checkpoints/epoch=233-step=351233-val_ExpRate=0.6284.ckpt'

ckpt = '/workspace/nemo/data/RESEARCH/HMER_CoMER/CoMER/epoch=279-step=313599-val_ExpRate=0.6030.ckpt'



model = LitCoMER.load_from_checkpoint(ckpt)
print("Model Loading Successfull.....!")
device = torch.device("cpu")
model = model.to(device)
model.eval()

data_path = "/workspace/nemo/data/RESEARCH/HMER_CoMER/CoMER/data/"
# def eval_on_image(img):

#     print("Prediction Starts.....!")
#     #model.eval()
#     #model = model.to(device)
#     img = ToTensor()(img)
#     mask = torch.zeros_like(img, dtype=torch.bool)
#     hyp = model.approximate_joint_search(img.unsqueeze(0), mask)[0]
#     pred_latex = vocab.indices2label(hyp.seq)

#     return pred_latex
def count_mismatches(pred, caption):
    pred_len = len(pred)
    caption_len = len(caption)
    min_len = min(pred_len, caption_len)
    mismatches = abs(pred_len - caption_len)  # initial mismatches due to length differences

    for i in range(min_len):
        if pred[i] != caption[i]:
            mismatches += 1

    return mismatches

test_year = ["2014","2016","2019"]
print("Evaluation starting..")
for y in test_year:
    Correct_Pred = 0
    One_Mismatch_Pred = 0
    Two_Mismatch_Pred  = 0
    img_path = data_path+y+"/img"
    caption_file = data_path+y+"/caption.txt"

    #Reading The Caption File
    df = pd.read_csv(caption_file, sep='\t', header=None, names=['img_name', 'caption'])

    count=0
    for images in os.listdir(img_path):
        name = images.replace('.bmp', '')
        img = Image.open(data_path+y+"/img/"+images)
        img = ToTensor()(img)
        mask = torch.zeros_like(img, dtype=torch.bool)
        hyp = model.approximate_joint_search(img.unsqueeze(0), mask)[0]
        pred = vocab.indices2label(hyp.seq)
        
        #pred = eval_on_image(img)
        caption = df.loc[df['img_name'] == name, 'caption'].values
        count=count+1
        mismatches = count_mismatches(pred, caption)
        #print("Prediction.", pred, count)
        if (pred == caption):
            Correct_Pred += 1
            One_Mismatch_Pred += 1
            Two_Mismatch_Pred += 1
            print("Match", images)
        if (mismatches == 1):
            print("One Missmatch->", images)
            print("caption->",caption)
            print("Pred->",pred)
            One_Mismatch_Pred += 1
            Two_Mismatch_Pred += 1
            
        if (mismatches == 2):
            print("Two Missmatch->", images)
            print("caption->",caption)
            print("Pred->",pred)
            Two_Mismatch_Pred += 1
            
            
            
    Exp_Rate = Correct_Pred/(len(df))

    #print(f"Expression Rate of CROHME {y} Test Set ==============> {Exp_Rate}")
    result = f"Expression Rate of CROHME {y} Test Set ==============> {Exp_Rate}\n"
    print(result)
    
    # Writing the result to a file in append mode
    with open("results.txt", "a") as file:
        file.write(result)










