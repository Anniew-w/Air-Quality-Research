import os
import torch
from model import FrePatchTST3_flashformer, FrePatchTST3_flatten, FrePatchTST3_Lsoftmax, FrePatchTST3_informer, \
    FrePatchTST3_flowformer, FrePatchTST3_real_flashformer, FrePatchTST3_reformer, FrePatchTST3_temporal, \
    FrePatchTST3_linear_fre, FrePatchTST3_linear_cha, \
    FrePatchTST3_fre_nopatch, FrePatchTST3_fre_patch, FrePatchTST3_attn_ablation, \
    FrePatchTST3_Lsoftmax_temporal, FrePatchTST3_mamba, FrePatchTST3_Laser, \
    FrePatchTST3_attn_shared, FrePatchTST3_attn_concat, FrePatchTST3_attn_patch, FrePatchTST3_fre_all, \
    FrePatchTST3_noshortcut, FrePatchTST3_attn_no_revin


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'FrePatchTST3_temporal': FrePatchTST3_temporal,
            'FrePatchTST3_flashformer': FrePatchTST3_flashformer,
            'FrePatchTST3_flatten': FrePatchTST3_flatten,
            'FrePatchTST3_Lsoftmax': FrePatchTST3_Lsoftmax,
            'FrePatchTST3_informer': FrePatchTST3_informer,
            'FrePatchTST3_reformer': FrePatchTST3_reformer,
            'FrePatchTST3_flowformer': FrePatchTST3_flowformer,
            'FrePatchTST3_real_flashformer': FrePatchTST3_real_flashformer,
            'FrePatchTST3_mamba': FrePatchTST3_mamba,
            'FrePatchTST3_Laser': FrePatchTST3_Laser,
            'FrePatchTST3_Lsoftmax_temporal': FrePatchTST3_Lsoftmax_temporal,
            'FrePatchTST3_linear_fre': FrePatchTST3_linear_fre,
            'FrePatchTST3_linear_cha': FrePatchTST3_linear_cha,
            'FrePatchTST3_fre_nopatch': FrePatchTST3_fre_nopatch,
            'FrePatchTST3_fre_patch': FrePatchTST3_fre_patch,
            'FrePatchTST3_attn_patch': FrePatchTST3_attn_patch,
            'FrePatchTST3_fre_all': FrePatchTST3_fre_all,
            'FrePatchTST3_noshortcut': FrePatchTST3_noshortcut,
            'FrePatchTST3_attn_ablation': FrePatchTST3_attn_ablation,
            'FrePatchTST3_attn_shared': FrePatchTST3_attn_shared,
            'FrePatchTST3_attn_concat': FrePatchTST3_attn_concat,
            'FrePatchTST3_attn_no_revin': FrePatchTST3_attn_no_revin,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
