'''
the model for learning the initial guess
'''

import os
from argparse import Namespace

import torch

from . import base_model, resnet
from ..utils import utils


class InitialGuesserFactory(object):
    @staticmethod
    def get_initial_guesser(opt):
        if opt.guess_model == 'init_guess':
            model = InitialGuesser(opt)
            model = utils.set_model_device(model)
        else:
            raise ValueError('unknown initial guess model:',
                             opt.loss_surface_name)
        return model


class InitialGuesser(base_model.BaseModel, torch.nn.Module):
    '''
    Model for learning the initial guess
    '''

    def __init__(self, opt):
        self.opt = opt
        self.name = 'init_guess'
        self.check_options()
        super(InitialGuesser, self).__init__()
        self.create_model()

    def check_options(self):
        if self.opt.guess_model != self.name:
            content_list = []
            content_list += ['You are not using the correct class for training or eval']
            utils.print_notification(content_list, 'ERROR')
            exit(1)

    def create_model(self):
        # Çıkış boyutu 8 (homografi parametreleri için)
        self.out_dim = 8
        # RGB giriş (3 kanal)
        self.input_features = 3
        
        # ResNet-18 konfigürasyonu
        resnet_config = self.create_resnet_config()
        
        # ResNet-18 modeli oluştur
        self.feature_extractor = resnet.resnet18(
            resnet_config,
            pretrained=False,  # Futbol sahası için pretrained kullanmıyoruz
            num_classes=self.out_dim,  # 8 boyutlu çıkış
            input_features=self.input_features,  # 3 kanal giriş
            zero_init_residual=True  # ResNet-18'in önerilen başlangıç ayarı
        )
        
        # Son katmanı kontrol et
        assert self.feature_extractor.fc.weight.shape == (8, 512), "Son katman boyutu yanlış!"
        assert self.feature_extractor.fc.bias.shape == (8,), "Bias boyutu yanlış!"
        
        if (hasattr(self.opt, 'load_weights_upstream') and self.opt.load_weights_upstream):
            assert resnet_config.pretrained is False, 'pretrained weights or imagenet weights'
            self.load_pretrained_weights()

    def create_resnet_config(self):
        # ResNet-18 için temel konfigürasyon
        need_spectral_norm = False  
        pretrained = False  # Pretrained kullanmıyoruz
        group_norm = 0  # Batch norm kullanıyoruz
        
        # Opsiyonel ayarlar
        if hasattr(self.opt, 'need_spectral_norm') and self.opt.need_spectral_norm:
            need_spectral_norm = self.opt.need_spectral_norm
        elif hasattr(self.opt, 'need_spectral_norm_upstream') and self.opt.need_spectral_norm_upstream:
            need_spectral_norm = self.opt.need_spectral_norm_upstream
            
        if hasattr(self.opt, 'group_norm'):
            group_norm = self.opt.group_norm
        elif hasattr(self.opt, 'group_norm_upstream'):
            group_norm = self.opt.group_norm_upstream
            
        if hasattr(self.opt, 'imagenet_pretrain') and self.opt.imagenet_pretrain:
            pretrained = True
            
        # ResNet konfigürasyonu
        resnet_config = Namespace(
            need_spectral_norm=need_spectral_norm,
            pretrained=pretrained,
            group_norm=group_norm
        )
        
        self.print_resnet_config(resnet_config)
        return resnet_config

    def forward(self, x):
        # Giriş kontrolü
        assert x.shape[1] == self.input_features, f"Giriş kanal sayısı {x.shape[1]}, beklenen {self.input_features}"
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Çıkış kontrolü
        assert features.shape[1] == self.out_dim, f"Çıkış boyutu {features.shape[1]}, beklenen {self.out_dim}"
        
        return features

    def load_pretrained_weights(self):
        '''load pretrained weights
        this function can load weights from another model.
        '''
        super().load_trained_weights()

    def _verify_checkpoint(self, check_options):
        pass

    def _get_checkpoint_path(self):
        checkpoint_path = os.path.join(self.opt.out_dir, self.opt.load_weights_upstream, 'checkpoint.pth.tar')
        return checkpoint_path
