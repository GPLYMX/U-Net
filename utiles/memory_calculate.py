# -*- coding: utf-8 -*-
# @Time : 2023/10/31 9:29
# @Author : GuoPeng

#--------------------------------------------------------------------#
#作用：计算模型需要占用的显存，方便知道显卡够不够用
#使用方法：将模型初始化之后，传入Calculate_gpu_memory()即可
#--------------------------------------------------------------------#

import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchstat import stat
from torchsummary import summary
from thop import profile



from models.network import AttU_Net

def Calculate_gpu_memory(Model,train_batch_size,img_wide,img_height):
    print("----------------计算模型要占用的显存------------")
    #step1#------------------------------------------------------------------计算模型参数占用的显存
    type_size = 4 #因为参数是float32,也就是4B
    para = sum([np.prod(list(p.size())) for p in Model.parameters()])
    print("Model {}:params:{:4f}M".format(Model._get_name(),para * type_size/1000/1000))
    #step2#------------------------------------------------------------------------计算模型的中间变量会占用的显存
    input = torch.ones((train_batch_size, 3, img_wide, img_height))
    input.requires_grad_(requires_grad=False)
    #遍历模型的每一个网络层（注意：一般模型都是嵌套建立的，这里只考虑了小于等于2层嵌套结构）
    mods = list(Model.named_children())
    out_sizes = []
    for i in range(0, len(mods)):
            mod = list(mods[i][1].named_children())
            if mod != []:
                for j in range(0, len(mod)):
                    m = mod[j][1]
                    #注意这里，如果relu激活函数是inplace则不用计算
                    if isinstance(m,nn.ReLU):
                        if m.inplace:
                            continue
                    print("网络层(不包括池化层,inplace为True的激活函数)：",m)
                    try: #一般不会把展平操作记录到里面去，因为没有在__init__中初始化，所以这里需要加上，如果不加上，将不能继续计算
                        out = m(input)
                    except RuntimeError:
                        input = torch.flatten(input, 1)
                        out = m(input)
                    out_sizes.append(np.array(out.size()))
                    if mod[j][0] not in ["rpn_score","rpn_loc"]:
                        input = out
            else:
                m = mods[i][1]
                #注意这里，如果relu激活函数是inplace则不用计算
                if isinstance(m,nn.ReLU):
                    if m.inplace:
                        continue
                print("网络层(不包括池化层,inplace为True的激活函数)：",m)
                try:
                    out = m(input)
                except RuntimeError:
                    input = torch.flatten(input, 1)
                    out = m(input)
                out_sizes.append(np.array(out.size()))

                # if mods[j][0] not in ["rpn_score","rpn_loc"]:
                #     input = out
    #统计每一层网络中间变量需要占用的显存
    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums
    print('Model {} : intermedite variables: {:3f} M (without backward)'
            .format(Model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
            .format(Model._get_name(), total_nums * type_size*2 / 1000 / 1000))
    print("----------------显存计算完毕------------")


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttU_Net(img_ch=13, output_ch=3).to(device)
    summary(model, input_size=(13, 656, 800))


    # stat(model, (13, 656, 800))
    # Calculate_gpu_memory(model,13,656,800)

    # input = torch.ones([1, 13, 656, 800]).cuda()
    # inputs = []
    # inputs.append(input)
    # flops, params = profile(net, inputs)  # ,custom_ops={model.ResNet,countModel})
    # print("flops:{0:,} ".format(flops))
    # print("parms:{0:,}".format(params))

