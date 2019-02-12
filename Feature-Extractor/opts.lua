--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet Training script')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '/mv_users/peiguo/dataset/cub-fewshot/full',         'Path to dataset')
   cmd:option('-dataset',    'in',       'Options: imagenet | cifar10 | cifar100')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-GPU',                1,  'Default preferred GPU')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        2,     'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         50,       'Number of total epochs to run')
   cmd:option('-decayEpoch',      30,       'Weight decay after ... epochs')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       16,       'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-testAll',         'false', 'Run on validation set for all trained models')
   cmd:option('-testBest',        'false', 'Run on validation set on best model')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-saveOutput',      'true',       'whether to save final layer')
   cmd:option('-logs',            'logs',        'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.001,   'initial learning rate')
   cmd:option('-momentum',        0.9,   'momentum')
   cmd:option('-weightDecay',     1e-4,  'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'resnet', 'Options: resnet | preresnet')
   cmd:option('-depth',        34,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-shortcutType', '',       'Options: A | B | C')
   cmd:option('-retrain',      '../models/resnet-50.t7',   'Path to model to retrain with')
   cmd:option('-rescbp',       'none',   'Path to model of resnet cbp') -- add cbp layer on top of resnet
   cmd:option('-resbp',        'none',   'Path to model of resnet bp') -- add bp layer on top of resnet. Inversed.
   cmd:option('-poof',         'none',   'Path to model of resnet poof')
   cmd:option('-respair',      'none',   'Path to model of resnet respair')
   cmd:option('-cbppair',      'none',   'Path to model of resnet cbppair')
   cmd:option('-siamese',      'none',   'Path to model of resnet siamese')
   cmd:option('-concat',       'none',   'Path to model of resnet but will do something inside ')
   cmd:option('-narrow',       'none',   'Path to model of resnet but will do something inside ')
   cmd:option('-dropout',      'none',   'Path to model of resnet to do targeted maskout ')
   cmd:option('-patches',      'none',   'Path to model of resnet but will do something inside ')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   cmd:option('-finetune',     'false',  'freeze layers below linear layers')
   cmd:option('-debug',        'false',  'print out inner parameters')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'true', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         200,    'Number of classes in the dataset')
   cmd:option('-pairNum',          1,      'Pairs of key points per image')
   cmd:option('-patchNum',         0,      'number of input image patches')
   cmd:option('-patchTotal',       105,    'how many patch in MLP input')      
   cmd:option('-patchSelect',      'false','indicator that only a few patches are selected ')
   cmd:option('-testIter',         1,      'how often the test occurs')
   cmd:option('-dispIter',         1,      'how often displaying')
   cmd:option('-slice',            1,      'slice the wingstack into 3 channel image')
   cmd:option('-sliceStart',       1,      'select the first slice to include in the input')
   cmd:option('-iterSize',         1,      'accumulate gradient after iterSize batches')
   cmd:option('-wingStack',        5,      'how many images in an input stack')
   cmd:option('-scale',            256,      'how many images in an input stack')
   cmd:option('-crop',             224,      'how many images in an input stack')
   cmd:option('-mlp',              'none', 'mlp model for weighting the models')
   cmd:option('-mlpbp',            'none', 'mlp + bp model')
   cmd:option('-visi',             'false','adding visibility to mlp')
   cmd:option('-conv1d',           'none', '1d convolution network')
   cmd:option('-mlpsia',           'false','using siamese network for mlp')
   cmd:option('-reweight',         'none', 'simplest gated network')
   cmd:option('-weighted',         'none', 'gated network with extra loss')
   cmd:option('-patchbp',          'none', 'use bilinear pooling')
   cmd:option('-restore',          'false','restore gradient descend')
   cmd:option('-unbalance',        'false','unbalance weighting')
   cmd:option('-maxkpool',         1,      'max pooling indices')
   cmd:option('-entropyWt',        1e-3,   'weight for entropy loss')
   cmd:text()

   local opt = cmd:parse(arg or {})
    
   --opt.patchNum = 1
   --opt.respair = '../models/resnet-50.t7'
   --opt.batchSize = 4
   --opt.LR = 0.001
   --opt.finetune = 'false'
   --opt.weightDecay = 1e-4
   --opt.debug = 'true'
   --opt.testOnly = 'true'
   --opt.dataset = 'onepatch'
   --opt.data = string.format('/mv_users/peiguo/dataset/patches/patch-%d/', opt.patchNum)
   --opt.resume = string.format('/mv_users/peiguo/snapshots/patches/checkpoint-%d/', opt.patchNum)
   --opt.save = string.format('./checkpoint-%d', opt.patchNum)
   --[[opt.retrain = '../models/th-vgg-16.t7'
   opt.batchSize = 8
   opt.LR = 0.001
   opt.finetune = 'false'
   opt.weightDecay = 1e-4
   opt.debug = 'false'
   opt.resetClassifier = 'true'
   opt.dataset = 'in'
   opt.data = '/home/peiguo/dataset/bf-256/'
   opt.nThreads = 10 
    ]]
   
   
   opt.timeString = os.date('%y_%m_%d_%H_%M_%S')

   -- create log file
   if not paths.dir(opt.logs) then paths.mkdir(opt.logs) end
   cmd:log(paths.concat(opt.logs, 'cmd_log_' .. opt.timeString), opt)
 
   opt.visi = opt.visi ~= 'false'
   opt.patchSelect = opt.patchSelect ~= 'false'
   opt.saveOutput = opt.saveOutput ~= 'false'
   opt.unbalance = opt.unbalance ~= 'false'
   opt.restore = opt.restore ~= 'false'
   opt.mlpsia = opt.mlpsia ~= 'false'
   opt.debug = opt.debug ~= 'false'
   opt.finetune = opt.finetune ~= 'false'
   opt.testBest = opt.testBest ~= 'false'
   opt.testOnly = opt.testOnly ~= 'false'
   opt.testAll  = opt.testAll  ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   -- Handle the most common case of missing -data flag
   local trainDir = paths.concat(opt.data, 'train')
   if not paths.dirp(opt.data) then
      cmd:error('error: missing ImageNet data directory')
   elseif not paths.dirp(trainDir) and opt.dataset ~= 'inat' then
      cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
   end
      
    
   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.CudaTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.CudaDoubleTensor'
   elseif opt.precision == 'half' then
      opt.tensorType = 'torch.CudaHalfTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
