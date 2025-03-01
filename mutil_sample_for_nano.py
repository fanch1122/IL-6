import os.path
import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from abnumber import Chain
from Bio import SeqIO
from nanosample import (
    batch_input_element, save_nano, seqs_to_fasta,
    get_multi_model_state, get_new_log_dir, get_logger
)
from utils.tokenizer import Tokenizer
from utils.train_utils import model_selected
from model.nanoencoder.abnativ_model import AbNatiV_Model
from model.nanoencoder.model import NanoAntiTFNet

def get_nano_seqs_from_fasta(fpath):
    """
    获取FASTA文件中的所有纳米抗体序列
    :param fpath: fasta文件路径
    :return: 纳米抗体序列列表
    """
    nano_chains = []
    sequences = SeqIO.parse(fpath, 'fasta')
    for seq in sequences:
        if 'Nanobody' in seq.description:
            nano_chains.append(str(seq.seq))
    assert nano_chains, "没有读取到任何纳米抗体序列！"
    return nano_chains

 # 打印读取到的所有纳米抗体序列
    print("读取到的纳米抗体序列：")
    for i, chain in enumerate(nano_chains):
        print(f"序列 {i+1}: {chain}")

    assert nano_chains, "没有读取到任何纳米抗体序列！"
    return nano_chain

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program is designed to humanize non-human nanobodies.")
    parser.add_argument('--ckpt', type=str, required=True, help='The ckpt path of the pretrained path.')
    parser.add_argument('--nano_complex_fasta', type=str, required=True, help='fasta file of the nanobody.')
    parser.add_argument('--batch_size', type=int, default=10, help='the batch size of sample.')
    parser.add_argument('--sample_number', type=int, default=100, help='The number of all sample.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_order', type=str, default='shuffle')
    parser.add_argument('--sample_method', type=str, default='gen', choices=['gen', 'rl_gen'])
    parser.add_argument('--length_limit', type=str, default='not_equal')
    parser.add_argument('--model', type=str, default='finetune_vh', choices=['pretrain', 'finetune_vh'])
    parser.add_argument('--fa_version', type=str, default='v_nano')
    parser.add_argument('--inpaint_sample', type=eval, default=True)
    parser.add_argument('--structure', type=eval, default=False)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取FASTA文件中的所有纳米抗体序列
    nano_chains = get_nano_seqs_from_fasta(args.nano_complex_fasta)

    # 日志和保存路径设置
    log_path = os.path.dirname(args.nano_complex_fasta)
    sample_tag = f'{os.path.basename(args.nano_complex_fasta).split(".")[0]}_{args.model}_vhh'
    log_dir = get_new_log_dir(root=log_path, prefix=sample_tag)
    logger = get_logger('test', log_dir)

    # 加载模型
    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    abnativ_state, _, infilling_state = get_multi_model_state(ckpt)

    abnativ_model = AbNatiV_Model(ckpt['abnativ_params'])
    abnativ_model.load_state_dict(abnativ_state)
    abnativ_model.to(device)

    infilling_model = NanoAntiTFNet(**ckpt['infilling_params'])
    infilling_model.load_state_dict(infilling_state)
    infilling_model.to(device)

    # Carefull!!! tmp
    config.model['equal_weight'] = True
    config.model['vhh_nativeness'] = False
    config.model['human_threshold'] = None
    config.model['human_all_seq'] = False
    config.model['temperature'] = False
    
    model_dict = {
        'abnativ': abnativ_model, 
        'infilling': infilling_model, 
        'target_infilling': infilling_model
    }
    framework_model = model_selected(config, pretrained_model=model_dict, tokenizer=Tokenizer())
    model = framework_model.infilling_pretrain
    model.eval()

    # 结果文件路径
    save_fpath = os.path.join(log_dir, 'sample_humanization_result.csv')
    with open(save_fpath, 'a', encoding='UTF-8') as f:
        f.write('Specific,name,hseq,\n')

    # 样本生成
    wrong_idx_list = []
    length_not_equal_list = []
    sample_number = args.sample_number

    # 批量处理每个纳米抗体序列
    for nano_chain in tqdm(nano_chains, desc="Humanizing nanobodies"):
        logger.info(f"正在人源化的序列：{nano_chain}")  # 显示当前处理的序列
        try:
            nano_pad_token, nano_pad_region, nano_loc, ms_tokenizer = batch_input_element(
                nano_chain, inpaint_sample=args.inpaint_sample, batch_size=args.batch_size
            )
        except Exception as e:
            logger.error(f"错误在处理序列 {nano_chain}: {str(e)}")
            continue

        if args.sample_order == 'shuffle':
            np.random.shuffle(nano_loc)

        duplicated_set = set()

        while sample_number > 0:
            all_token = ms_tokenizer.toks
            with torch.no_grad():
                for i in tqdm(nano_loc, total=len(nano_loc), desc='Nanobody Humanization Process'):
                    nano_prediction = model(
                        nano_pad_token.to(device),
                        nano_pad_region.to(device),
                        H_chn_type=None
                    )

                    nano_pred = nano_prediction[:, i, :len(all_token)-1]
                    nano_soft = torch.nn.functional.softmax(nano_pred, dim=1)
                    nano_sample = torch.multinomial(nano_soft, num_samples=1)
                    nano_pad_token[:, i] = nano_sample.squeeze()

            nano_untokenized = [ms_tokenizer.idx2seq(s) for s in nano_pad_token]
            for _, g_h in enumerate(nano_untokenized):
                if sample_number == 0:
                    break

                with open(save_fpath, 'a', encoding='UTF-8') as f:
                    sample_origin = 'humanization'
                    sample_name = str(nano_chain)
                    if g_h not in duplicated_set:
                        test_chain = Chain(g_h, scheme='imgt')
                        f.write(f'{sample_origin},{sample_name},{g_h}\n')
                        duplicated_set.add(g_h)
                        sample_number -= 1
                        logger.info(f"已生成样本 {args.sample_number - sample_number}：{g_h}")
                    else:
                        sample_number -= 1

    # 保存为FASTA格式文件
    fasta_save_fpath = os.path.join(log_dir, 'sample_identity.fa')
    logger.info(f'保存FASTA文件: {fasta_save_fpath}')
    sample_df = pd.read_csv(save_fpath)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    seqs_to_fasta(sample_human_df, fasta_save_fpath, version=args.fa_version)

    # 如果需要，分割保存为结构预测用的FASTA文件
    if args.structure:
        split_fasta_for_save(save_fpath)

    logger.info(f"生成过程完成，结果已保存至：{save_fpath}")
