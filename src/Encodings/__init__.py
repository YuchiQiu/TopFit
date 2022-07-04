from Encodings import encodings
from Encodings.encodings import JointEncoder


BASIC_ENCODER_MAP = {
    'onehot': encodings.Onehot,
    'georgiev': encodings.Georgiev,
    'eunirep_pll': encodings.EUniRepPLL,
    'esm1b_pll': encodings.ESMPLL,
    'esm1v_pll': encodings.ESMPLL,
    'vae': encodings.VaeElbo,
}

def get_encoder_cls(encoder_name):
    names = encoder_name.split('+')
    encoder_cls=[]
    for n in names:
        if n in BASIC_ENCODER_MAP:
            encoder_cls.append(BASIC_ENCODER_MAP[n])
        else:
            encoder_cls.append(encodings.LoadNumpy)
    return encoder_cls

def get_encoder_names(key):

    return [key]
