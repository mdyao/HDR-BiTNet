import logging
from models.modules.BiT_arch import *
from models.modules.Subnet_constructor import subnet
import math
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    subnet_type_sgt = which_model['subnet_type_sgt']

    down_num = int(math.log(opt_net['scale'], 2))
    netG = InvNet( subnet(subnet_type), subnet(subnet_type_sgt),  down_num)

    return netG
