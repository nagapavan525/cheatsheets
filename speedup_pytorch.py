# FastAI kernel scores better than Pytorch kernel as fastai learner used AdamW by default wd of 0.01.

# https://www.fast.ai/2018/07/02/adam-weight-decay/

# According to that blog, just need to add 3 lines during the training phase when using PyTorch to use FastAI version AdamW

for group in optimizer.param_groups():
    for param in group['params']:
        param.data = param.data.add(-wd * group['lr'], param.data)
