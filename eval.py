import pdb
import random
from data import FasterImageDetectionsField, TextField, RawField
from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer
from evaluation.cider import Cider
import json
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import time
from data import build_image_field
from models import model_factory
# from line_profiler import LineProfiler
from contiguous_params import ContiguousParams
import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def evaluate_metrics(model, dataloader, text_field,cider):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    res = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            caps_gen1 = text_field.decode(out)
            caps_gt1 = list(itertools.chain(*([c, ] * 1 for c in caps_gt)))
            caps_gen1, caps_gt1 = map(evaluation.PTBTokenizer.tokenize, [caps_gen1, caps_gt1])
            reward = cider.compute_score(caps_gt1, caps_gen1)[1].astype(np.float32)
            for i,(gts_i, gen_i) in enumerate(zip(caps_gt1,caps_gen1)):
                res[len(res)] = {
                    'gt':caps_gt1[gts_i],
                    'gen':caps_gen1[gen_i],
                    'cider':reward[i].item(),
                }
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()


    gts = evaluation.PTBTokenizer.tokenize(gts)  # 这里做没啥问题，因为多轮Tokenize在验证/测试集上没影响
    gen = evaluation.PTBTokenizer.tokenize(gen)  #
    #print('examples:')
    #print('gen:', gen['0_0'])
    #print('gt:', gts['0_0'])
    scores, _ = evaluation.compute_scores(gts, gen)

    # json.dump(res,open('pred_test.json','w'))
    return scores



#python train.py --exp_name test --batch_size 50 --head 8 --features_path ~/datassd/coco_detections.hdf5 --annotation_folder annotation --workers 8 --rl_batch_size 100 --image_field FasterImageDetectionsField --model transformer --seed 118
device = torch.device('cuda:2')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--features_path', type=str,default='/data16/wxx/image-captioning-DLCT-main/coco_all_align.hdf5')
    parser.add_argument('--annotation_folder', type=str,default='/data16/wxx/ORAT_Collection/Better_than_m2/annotation')
    parser.add_argument('--model_path', type=str,default='saved_models/test_best_test.pth')
    args = parser.parse_args()

    random.seed(118)
    torch.manual_seed(118)
    np.random.seed(118)

    print('Transformer Training')


    # Pipeline for image regions

    image_field = FasterImageDetectionsField(detections_path=args.features_path, max_detections=49,load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='split',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits


    text_field.vocab = pickle.load(open('.vocab_cache/vocab_test.pkl', 'rb'))


    start = time.time()
    # Model and dataloaders
    # Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention = model_factory(args)
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)


    data = torch.load(args.model_path,map_location={'cuda:2':'cuda:1'})
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size,num_workers=args.workers)

    ref_caps_test = list(test_dataset.text)
    cider_test = Cider(PTBTokenizer.tokenize(ref_caps_test))
    scores = evaluate_metrics(model, dict_dataloader_test, text_field,cider_test)
    print(scores)