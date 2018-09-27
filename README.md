# image-caption
简单的image caption的实现
trian:
  保证本目录下的data/mscoco文件加中已经有data.p dataset.json resnet_feats.map等文件。训练好的模型保存在./log中。
  直接执行python train.py启动训练
evaluate:
  该程序会使用5k的测试集来测试训练好的模型，程序会生成5k的captions，保存在coco_5k_test.txt中。
  执行python eva_script.py来启动测试。
评估：
   我们使用了SCN的测试代码来生成BLUE-4等评估结果，执行python SCN_evalution.py后，程序会自动比对coco_5k_text.txt中captions与      dataset.json中captions，生成包括BLUE-4在内的多种评估结果。

configuration.py中是模型和训练的参数
