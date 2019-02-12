--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
--require 'tcbp'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'gnuplot'

local M = {}

function M.setup(opt, checkpoint)
    local model
    if checkpoint then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model = torch.load(modelPath):cuda()
    elseif opt.retrain ~= 'none' then
        assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
        print('Loading model from file: ' .. opt.retrain)
        model = torch.load(opt.retrain):cuda()
        model.__memoryOptimized = nil
    elseif opt.narrow ~= 'none' then
        assert(paths.filep(opt.narrow), 'File not found: ' .. opt.narrow)
        print('Loading model from file: ' .. opt.narrow)
        model = torch.load(opt.narrow)
        --model:remove(11)
        --model:remove(10)
        --model:insert(nn.Narrow(2, 1, 512), 9)
        --model:add(nn.View(512))
        --model:add(nn.Linear(512, opt.nClasses))
        model:add(nn.Dropout(0.75), 10)
        
        model = model:cuda()
        model.__memoryOptimized = nil
    elseif opt.resbp ~= 'none' then
        assert(paths.filep(opt.resbp), 'File not found: ' .. opt.resbp)
        print('Loading model from file: ' .. opt.resbp)
        model = torch.load(opt.resbp)
        model:remove(#model.modules)
        model:remove(#model.modules)

        local featsize = 7 * 7
        local featdim = 2048
        local concat = nn.ConcatTable()
        concat:add(nn.View(-1, featdim, featsize))
        concat:add(nn.View(-1, featdim, featsize))
        model:add(concat)
        model:add(nn.MM(true , false))
        model:add(nn.View(-1, featsize))
        model:add(nn.Normalize(2))

        local l1 = nn.Linear(featsize, featdim)
        l1.bias:zero()
        model:add(l1)
        local l2 = nn.Linear(featdim, 200)
        l2.bias:zero()
        model:add(l2)

        --weight initialization
        --model:get(#model.modules):reset()

        model = model:cuda()
        model.__memoryOptimized = nil
    elseif opt.rescbp ~= 'none' then
        assert(paths.filep(opt.rescbp), 'File not found: ' .. opt.rescbp)
        print('Loading model from file: ' .. opt.rescbp)
        model = torch.load(opt.rescbp)
        model:remove(#model.modules)
        model:remove(#model.modules)

        model:add(nn.ComBiPooling(8192, true))
        model:add(nn.SignedSquareRoot())
        model:add(nn.Normalize(2))

        local newLinear = nn.Linear(8192, 200)
        newLinear.bias:zero()

        model:add(newLinear)

        --weight initialization
        model:get(#model.modules):reset()

        model = model:cuda()
        model.__memoryOptimized = nil
    elseif opt.poof ~= 'none' then
        assert(paths.filep(opt.poof), 'File not found: ' .. opt.poof)
        print('Loading model from file: ' .. opt.poof)

        local generator = torch.load(opt.poof)
        generator:remove(#generator)
        generator:add(nn.Squeeze())

        local bp = nn.Sequential()
        local bpConcat = nn.ConcatTable():add(nn.Identity()):add(nn.Identity())
        bp:add(bpConcat)
        bp:add(nn.MM(false, true))
        bp:add(nn.View(1, opt.pairNum * opt.pairNum))
        bp:add(nn.SignedSquareRoot())

        local cl = nn.Sequential()
        local ll = nn.Linear(opt.pairNum * opt.pairNum, 200)
        ll.bias:zero()
        ll:reset()
        cl:add(ll)

        model = nn.Sequential()
        model:add(generator):add(bp):add(cl)
        model = model:cuda()
        model.__memoryOptimized = nil

    elseif opt.respair ~= 'none' then
        assert(paths.filep(opt.respair), 'File not found: ' .. opt.respair)
        print('Loading model from file: ' .. opt.respair)
        model = torch.load(opt.respair)

        model:remove(#model.modules)
        model:remove(#model.modules)
        model:remove(#model.modules)

        model:add(nn.SpatialAveragePooling(4,2))
        model:add(nn.View(-1, opt.pairNum * 2048))
        local linear = nn.Linear(opt.pairNum * 2048, 200)
        linear.bias:zero()
        model:add(linear)

        model = model:cuda()
        model.__memoryOptimized = nil

    elseif opt.cbppair ~= 'none' then
        assert(paths.filep(opt.cbppair), 'File not found: ' .. opt.cbppair)
        print('Loading model from file: ' .. opt.cbppair)
        model = torch.load(opt.cbppair)
        --[[
        for i = 1, 9 do
            model:remove(#model.modules)
        end

        local container = nn.Sequential()
        local cbpConcat = nn.ConcatTable():add(nn.Identity()):add(nn.Identity())
        local cbp = nn.ComBiPooling(2048 * 4)
        local factor = nn.MulConstant(8192*8192)
        local ssr = nn.SignedSquareRoot()

        container:add(cbpConcat):add(cbp)
        container:add(factor)
        container:add(ssr)
        local nrm = nn.Normalize(2)
        container:add(nrm)

        model:add(container)
        local linear = nn.Linear(2048 * 4, 200)
        linear.bias:zero()
        model:add(linear)
        ]]
        model = model:cuda()
        model.__memoryOptimized = nil
    elseif opt.siamese ~= 'none' then
        assert(paths.filep(opt.siamese), 'File not found: ' .. opt.siamese)
        print('Loading model from file: ' .. opt.siamese)
        local resnet = torch.load(opt.siamese)

        local orig = resnet:get(#resnet.modules)
        assert(torch.type(orig) == 'nn.Linear',
        'expected last layer to be fully connected')

        local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
        linear.bias:zero()

        resnet:remove(#resnet.modules)
        resnet:add(linear)

        --local half_splitter = nn.Sequential()
        --half_splitter:add(nn.View(-1, opt.batchSize / 2, opt.nClasses))
        --half_splitter:add(nn.SplitTable(1, 2))

        local half_splitter = nn.HalveTable(1)

        local concat = nn.ConcatTable():add(nn.Identity()):add(half_splitter)
        model = nn.Sequential():add(resnet):add(concat)

        model = model:cuda()

    elseif opt.mlp ~= 'none' then
        local featdim = opt.nClasses
        local nhiddens = 1024
        local ninputs = featdim * opt.patchTotal 
        local noutputs = opt.nClasses

        local mlp = nn.Sequential()
        local stretch = nn.View(-1):setNumInputDims(2)
        mlp:add(stretch)

        local l1 = nn.Linear(ninputs, nhiddens)
        local l2 = nn.Linear(nhiddens, noutputs)

        mlp:add(l1)
        mlp:add(nn.BatchNormalization(nhiddens))
        mlp:add(nn.ReLU())
        mlp:add(l2)

        -- it's absolutely necessary to add LogSoftMax
        -- other code without this layer is suspected to be wrong.
        if opt.mlpsia then
            local concat = nn.ConcatTable():add(nn.Identity()):add(nn.LogSoftMax())
            model = nn.Sequential():add(mlp):add(concat)
        else
            model = mlp
        end

        model = model:cuda()
        model.__memoryOptimized = nil

    elseif opt.visi then
        local bsize = 105
        local psize = 200
        local ninputs = bsize * psize
        local nhiddens = 1024
        local ksize = 15
        local nclasses = 200

        local middle = - nn.Identity()
        local diag_left =  nn.Sequential():add(nn.Linear(ksize, bsize)):add(nn.Diagonal())
        local diag_right =  nn.Sequential():add(nn.Linear(ksize, psize)):add(nn.Diagonal())
        local diag_gen = - nn.ConcatTable():add(diag_left):add(diag_right)
        local left = diag_gen 
                     - nn.SelectTable(1)

        local right = diag_gen 
                      - nn.SelectTable(2)

        local mm1 = {left, middle}
                    - nn.MM()

        local mm2 = {mm1, right}
                    - nn.MM()

        local lin = mm2 
                    - nn.View(-1):setNumInputDims(2)
                    - nn.Linear(ninputs, nhiddens)
                    - nn.BatchNormalization(nhiddens)
                    - nn.ReLU()
                    - nn.Linear(nhiddens, nclasses)

        model = nn.gModule({middle, diag_gen}, {lin})

        model = model:cuda()
        model.__memoryOptimized = nil

    elseif opt.mlpbp ~= 'none' then
        local ninputs = 200 * 200
        local nhiddens = 2048
        local noutputs = 200
        local patchsize = 105
        local featsize = 200
        
        local bp = nn.Sequential()
        local norm = nn.Sequential()
        norm:add(nn.View(-1, featsize))
        norm:add(nn.Normalize(2))
        norm:add(nn.View(-1, patchsize, featsize))
        local bpConcat = nn.ConcatTable():add(nn.Identity()):add(nn.Identity())
        --bp:add(norm)
        bp:add(bpConcat)
        bp:add(nn.MM(true , false))
        bp:add(nn.View(-1, ninputs))
        --bp:add(nn.SignedSquareRoot())
        bp:add(nn.Normalize(2))
        --bp:add(nn.BatchNormalization(ninputs))
        --bp:add(nn.ReLU())

        local mlp = nn.Sequential()
        mlp:add(nn.Reshape(ninputs,true))

        local l1 = nn.Linear(ninputs, nhiddens)
        local l2 = nn.Linear(nhiddens, noutputs)

        mlp:add(l1)
        mlp:add(nn.BatchNormalization(nhiddens))
        mlp:add(nn.ReLU())
        mlp:add(l2)

        if opt.mlpsia then
            local half_splitter = nn.HalveTable(1)
            local concat = nn.ConcatTable():add(nn.Identity()):add(half_splitter)
            model = nn.Sequential():add(mlp):add(concat)
        else
            model = nn.Sequential():add(bp):add(mlp)
        end

        model = model:cuda()
        model.__memoryOptimized = nil

        model.__memoryOptimized = nil

    elseif opt.conv1d ~= 'none' then
        local featdim = 2048
        local patchdim = 105
        local outdim = 256 

        local noutputs = 200
        local nhiddens = outdim * patchdim

        local mlp = nn.Sequential()

        local tc = nn.TemporalConvolution(featdim, outdim, 1)
        local ll = nn.Linear(nhiddens, noutputs)

        mlp:add(tc)
        mlp:add(nn.View(-1, nhiddens))
        mlp:add(nn.BatchNormalization(nhiddens))
        mlp:add(nn.ReLU())
        mlp:add(ll)

        model = mlp

        model = model:cuda()
        model.__memoryOptimized = nil

    elseif opt.patches ~= 'none' then
        assert(paths.filep(opt.patches), 'File not found: ' .. opt.patches)
        print('Loading model from file: ' .. opt.patches)
        model = torch.load(opt.patches)

        local wingStack = opt.wingStack 
        local inputSize = {64, 128} 
        local outputSize = 2048

        model:insert(nn.SpatialAveragePooling(4,2,1,1), 9)
        model:remove(10)

        local shift = nn.View(3, inputSize[1], inputSize[2])
        model:insert(shift, 1)
        local concat = nn.View(outputSize*wingStack)
        model:remove(#model.modules-1)
        model:insert(concat, #model.modules)

        local linear = nn.Linear(outputSize*wingStack, opt.nClasses)
        linear.bias:zero()

        model:remove(#model.modules)
        model:add(linear)

        model = model:cuda()
        model.__memoryOptimized = nil
    elseif opt.concat ~= 'none' then
        assert(paths.filep(opt.concat), 'File not found: ' .. opt.concat)
        print('Loading model from file: ' .. opt.concat)
        model = torch.load(opt.concat)

        local wingStack = opt.wingStack 
        local inputSize = {224, 224} 
        local outputSize = 2048

        local shift = nn.View(-1, 3, inputSize[1], inputSize[2])
        model:insert(shift, 1)
        local concat = nn.View(outputSize*wingStack)
        model:remove(#model.modules-1)
        model:insert(concat, #model.modules)

        local linear = nn.Linear(outputSize*wingStack, opt.nClasses)
        linear.bias:zero()

        model:remove(#model.modules)
        model:add(linear)

        model = model:cuda()
        model.__memoryOptimized = nil
    elseif opt.dropout ~= 'none' then
        assert(paths.filep(opt.dropout), 'File not found: ' .. opt.dropout)
        print('Loading model from file: ' .. opt.dropout)
        local resnet = torch.load(opt.dropout)

        local wingStack = opt.wingStack 
        local inputSize = {224, 224} 
        local outputSize = 2048

        local shift = nn.View(-1, 3, inputSize[1], inputSize[2])
        resnet:insert(shift, 1)
        local concat = nn.View(outputSize*wingStack)
        --resnet:remove(#resnet.modules-1)
        --resnet:insert(concat, #resnet.modules)

        local linear = nn.Linear(outputSize*wingStack, opt.nClasses)
        linear.bias:zero()

        resnet:remove(#resnet.modules)
        --resnet:add(linear)

        local mask = nn.TargetedMask(outputSize)

        model = nn.Sequential()
        local ctable = nn.ConcatTable()
        ctable:add(mask)
        ctable:add(resnet)
        model:add(ctable)

        model:add(nn.CMulTable())
        
        model:add(concat)
        model:add(linear)

        model = model:cuda()
        model.__memoryOptimized = nil

    elseif opt.reweight ~= 'none' then
        model = nn.Sequential()

        local leftOp = nn.Transpose({2,3})

        local rightOp = nn.Sequential()
        rightOp:add(nn.View(-1, 105*200))
        local rlin = nn.Linear(105*200, 105)
        rightOp:add(nn.Linear(105*200, 1024))
        rightOp:add(nn.BatchNormalization(1024))
        rightOp:add(nn.ReLU())
        rightOp:add(nn.Linear(1024, 105))

        rightOp:add(nn.KMaxPooling(opt.maxkpool))
        rightOp:add(nn.SoftMax())
        rightOp:add(nn.View(-1, 105, 1))

        local prepTable = nn.ConcatTable()
        prepTable:add(leftOp):add(rightOp)

        model:add(prepTable)
        model:add(nn.MM())
        model:add(nn.View(-1, 200))

        model = model:cuda()
        model.__memoryOptimized = nil

    elseif opt.weighted ~= 'none' then
        local psize = opt.patchTotal

        local input = - nn.Identity()
        local trans = input
                      - nn.Transpose({2,3})

        local bweight = input
                        - nn.View(-1, 200*psize)
                        -- nn.Linear(105*200, 105)
                        - nn.Linear(200*psize, 1024)
                        - nn.ReLU()
                        - nn.BatchNormalization(1024)
                        - nn.Linear(1024, psize)
                        
        local logwt = bweight
                      - nn.LogSoftMax()

        local gated = bweight
                      - nn.KMaxPooling(opt.maxkpool)
                      - nn.SoftMax()
                      - nn.View(-1, psize, 1)

        local mm = {trans, gated} 
                   - nn.MM()
                   - nn.View(-1, 200)

        model = nn.gModule({input}, {mm, logwt})

        model = model:cuda()
        model.__memoryOptimized = nil

    elseif opt.patchbp ~= 'none' then
        model = nn.Sequential()

        local concat = nn.ConcatTable()
        concat:add(nn.Identity()):add(nn.Identity())

        model:add(concat)
        model:add(nn.MM(false, true))

        model:add(nn.View(-1, 105*105))
        model:add(nn.SignedSquareRoot())
        model:add(nn.Normalize(2))
        model:add(nn.Linear(105*105, 1024))
        model:add(nn.ReLU())
        model:add(nn.BatchNormalization(1024))
        model:add(nn.Linear(1024, 200))

        model = model:cuda()
        model.__memoryOptimized = nil

    end

    -- freeze all layers below cbp
    if opt.finetune then
        model:apply(function(m) 
            if torch.type(m):find("BatchNormalization") then 
                m:evaluate()
            end
            if torch.type(m):find("Convolution") then 
                --m.accBak = m.accGradParameters
                m.accGradParameters = function() end 
                --m:learningRate('weight', 0)
                --m:learningRate('bias', 0)
                --m:evaluate()
            end
        end) -- for freezing the parameters end)
    end
    if opt.restore then
        model:apply(function(m)
            if torch.type(m):find("BatchNormalization") then 
                m:training()
            end
            if torch.type(m):find("Convolution") then 
                m.accGradParameters = cudnn.SpatialConvolution.accGradParameters
            end
        end)
    end
    -- First remove any DataParallelTable
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    -- optnet is an general library for reducing memory usage in neural networks
    if opt.optnet then
        local optnet = require 'optnet'
        local imsize = opt.dataset == 'imagenet' and 224 or 32
        local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
        optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
    end

    -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
    -- containers override backwards to call backwards recursively on submodules
    if opt.shareGradInput then
        M.shareGradInput(model)
    end

    -- For resetting the classifier when fine-tuning on a different Dataset
    if opt.resetClassifier and not checkpoint then
        local gnuplot = require 'gnuplot'
        print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

        local orig = model:get(#model.modules)
        assert(torch.type(orig) == 'nn.Linear',
        'expected last layer to be fully connected')

        local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
        linear.bias:zero()

        model:remove(#model.modules)
        model:add(linear:cuda())

        model = model:cuda()
    end

    -- Set the CUDNN flags
    if opt.cudnn == 'fastest' then
        cudnn.fastest = true
        cudnn.benchmark = true
    elseif opt.cudnn == 'deterministic' then
        -- Use a deterministic convolution implementation
        model:apply(function(m)
            if m.setMode then m:setMode(1, 1, 1) end
        end)
    end

    -- Wrap the model with DataParallelTable, if using more than one GPU
    if opt.nGPU > 1 then
        local gpus = torch.range(1, opt.nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
            require 'tcbp'
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end

    local criterion = {} 
    if opt.siamese == 'none' and (not opt.mlpsia) and opt.weighted == 'none' then 
        criterion = nn.CrossEntropyCriterion():cuda()
    else
        local predict_crit = nn.CrossEntropyCriterion()
        local entropy_crit = nn.DistKLDivCriterion()
        criterion = nn.ParallelCriterion()
        criterion:add(predict_crit, 1)
        criterion:add(entropy_crit, opt.entropyWt)
        criterion = criterion:cuda()
    end  

    return model, criterion
end

function M.shareGradInput(model)
    local function sharingKey(m)
        local key = torch.type(m)
        if m.__shareGradInputKey then
            key = key .. ':' .. m.__shareGradInputKey
        end
        return key
    end

    -- Share gradInput for memory efficient backprop
    local cache = {}
    model:apply(function(m)
        local moduleType = torch.type(m)
        if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
            local key = sharingKey(m)
            if cache[key] == nil then
                cache[key] = torch.CudaStorage(1)
            end
            m.gradInput = torch.CudaTensor(cache[key], 1, 0)
        end
    end)
    for i, m in ipairs(model:findModules('nn.ConcatTable')) do
        if cache[i % 2] == nil then
            cache[i % 2] = torch.CudaStorage(1)
        end
        m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
    end
end

return M
