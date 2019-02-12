--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local io = require 'io'
local gnuplot = require 'gnuplot'
local image = require 'image'
local color = require 'color'
local image = require 'image'
local paths = require 'paths'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:memoryUsage()
    local Gig = 1024*1024*1024
    local free, total = cutorch.getMemoryUsage()
    print("free memory: " .. free/Gig .. "GB", "total memory: " .. total/Gig .. "GB") 
end

function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
    }
    self.opt = opt
    self.params, self.gradParams = model:getParameters()
    self.trainLogger = optim.Logger(paths.concat(opt.logs, 'optim_log_' .. opt.timeString))
end

function Trainer:train(epoch, dataloader)
    -- Trains the model for a single epoch
    self.optimState.learningRate = self:learningRate(epoch)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local trainSize = dataloader:size()
    local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
    local N = 0
    local innerIter = 0
    local peakStat = nil

    print('=> Training epoch # ' .. epoch)
    -- set the batch norm to training mode
    self.model:training()
    for n, sample in dataloader:run() do
        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)

        local output = self.model:forward(self.input)
        
        if self.opt.reweight ~= 'none' then
            local prob = self.model:get(1).output[2]:float()
            local pv, pi = prob:sort(2, true)
            local peak = pi:select(2,1):clone():view(-1)--, pv:select(2,1):clone():view(1,-1))
            if not peakStat then
                peakStat = torch.zeros(prob:size(2))
            end

            for _, v in ipairs(peak:totable()) do
                peakStat[v] = peakStat[v] + 1
            end

            gnuplot.figure(1)
            gnuplot.plot(peakStat, '|')
            --os.execute("sleep 1") 
        end

        if self.opt.weighted ~= 'none' then
            local prob = self.model.output[2]:float()
            local pv, pi = prob:sort(2, true)
            local peak = pi:select(2,1):clone():view(-1)--, pv:select(2,1):clone():view(1,-1))
            if not peakStat then
                peakStat = torch.zeros(prob:size(2))
            end

            for _, v in ipairs(peak:totable()) do
                peakStat[v] = peakStat[v] + 1
            end

            --gnuplot.figure(1)
            --gnuplot.plot(peakStat, '|')
            --os.execute("sleep 1") 
        end

        if self.opt.siamese ~= 'none' or self.opt.mlpsia or self.opt.weighted ~= 'none' then
            output = output[1]:float()
        else
            output = output:float()
        end


        local batchSize = output:size(1)
        local loss = self.criterion:forward(self.model.output, self.target)

        if innerIter == 0 then
            self.model:zeroGradParameters()
        end
        self.criterion:backward(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        innerIter = innerIter + 1
         
        if innerIter == self.opt.iterSize or n == trainSize then
            -- divide grads by innerIter
            if innerIter > 1 then
                self.gradParams:div(innerIter)
            end
            optim.sgd(feval, self.params, self.optimState)
            innerIter = 0
        end

        --optim.sgd(feval, self.params, self.optimState)

        local top1, top5 = self:computeScore(output, sample.target, 1)
        top1Sum = top1Sum + top1*batchSize
        top5Sum = top5Sum + top5*batchSize
        lossSum = lossSum + loss*batchSize
        N = N + batchSize

        if n % self.opt.dispIter == 0 then
            print((color.fg.red .. ' | Epoch: [%d][%d/%d]  LR %.1e  Time %.3f  Data %.3f  Err %1.4f(%1.4f)  top1 %7.3f(%7.3f)  top5 %7.3f(%7.3f)'):format(
            epoch, n, trainSize, self.optimState.learningRate, timer:time().real, dataTime, loss, lossSum/N, top1, top1Sum/N, top5, top5Sum/N))

            if self.opt.debug then
                local activation = self.model:get(9).output
                win1 = image.display({image = activation[1], win = win1, nrow = 64, zoom = 4})
                win2 = image.display({image = self.input[1], win = win2})
                image.save('../pictures/act/' .. paths.basename(sample.name[1]) .. '_train_act.png', win1.painter:image():toTensor())
                image.save('../pictures/act/' .. paths.basename(sample.name[1]) .. '_train.png', win2.painter:image():toTensor())
            end
        end

        -- check that the storage didn't get changed do to an unfortunate getParameters call
        assert(self.params:storage() == self.model:parameters()[1]:storage())

        timer:reset()
        dataTimer:reset()
    end
    
    
    if self.opt.reweight ~= 'none' and false then
        local _, rank_peak = peakStat:sort(1, true)
        local top_peak = rank_peak:narrow(1,1,self.opt.maxkpool)

        local save_name = '../ipynb/t7s/sparse-gate-stat.t7'
        local result_table = {}
        if not paths.filep(save_name) then
            result_table[self.opt.maxkpool] = {}
            result_table[self.opt.maxkpool].patch = top_peak
        else
            result_table = torch.load(save_name)
            result_table[self.opt.maxkpool] = {}
            result_table[self.opt.maxkpool].patch = top_peak
        end
        torch.save(save_name, result_table)
    end

    --self:memoryUsage()
    return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
    -- Computes the top-1 and top-5 err on the validation set


    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local size = dataloader:size()

    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
    local N = 0

    local conf = {}
    if self.opt.unbalance then
        conf = optim.ConfusionMatrix( {'American-Crow','Fish-Crow'} )   -- new matrix
        conf:zero()                                              -- reset matrix
    end

    self.model:evaluate()
    self.resultTable = {}
    for n, sample in dataloader:run() do
        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)
        --print(sample.name, sample.target)

        local output = self.model:forward(self.input)
        local target = self.target

        if self.opt.siamese ~= 'none' or self.opt.mlpsia or self.opt.weighted ~= 'none' then
            output = output[1]:float()
            target = target[1]:float()
        else
            output = output:float()
            target = target:float()
        end

        local batchSize = output:size(1) / nCrops
        local loss = self.criterion:forward(self.model.output, self.target)
        if self.opt.unbalance then
            conf:batchAdd(output, target)
        end

        -- for each input, save ground truth, last layer feature.
        if self.opt.saveOutput then
            self:saveResult(sample)
        end

        local top1, top5 = self:computeScore(output, sample.target, nCrops)
        top1Sum = top1Sum + top1*batchSize
        top5Sum = top5Sum + top5*batchSize
        lossSum = lossSum + loss*batchSize
        N = N + batchSize

        if n % self.opt.dispIter == 0 then
            print((color.fg.red .. ' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f(%1.4f)  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
            epoch, n, size, timer:time().real, dataTime, loss, lossSum/N, top1, top1Sum / N, top5, top5Sum / N))


            if self.opt.debug then
                local activation = self.model:get(9).output
                win1 = image.display({image = activation[1], win = win1, nrow = 64, zoom = 4})
                win2 = image.display({image = self.input[1], win = win2})
                image.save('../pictures/act/' .. paths.basename(sample.name[1]) .. '_test_act.png', win1.painter:image():toTensor())
                image.save('../pictures/act/' .. paths.basename(sample.name[1]) .. '_test.png', win2.painter:image():toTensor())
            end
        end

        timer:reset()
        dataTimer:reset()
    end
    self.model:training()

    if self.opt.unbalance then
        print(conf)                                              -- print matrix
        --win1 = image.display{conf:render(), win = win1}                             -- render matrix
        --sys.sleep(60)
    end
    print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
    epoch, top1Sum / N, top5Sum / N))

    if self.opt.saveOutput then
        local timestr 
        if self.opt.testOnly then
            timestr = paths.basename(self.opt.retrain):gmatch("%d[%d_]+")()
        else
            timestr = self.opt.timeString
        end
        torch.save(paths.concat(self.opt.save, 'feature_' .. dataloader.split .. '_' .. timestr .. '.t7'), self.resultTable)
    end
    return top1Sum / N, top5Sum / N, lossSum
end

function Trainer:saveResult(sample)
    local output = self.model:get(#self.model).output
    if output:nDimension() == 1 then
        output = output:view(1, output:size(1))
    end
    for i = 1, output:size(1) do
        local idx = #self.resultTable+1
        self.resultTable[idx] = {}
        self.resultTable[idx].name = sample.name[i]
        self.resultTable[idx].gt = sample.target[i]
        local feature = output:float()
        if feature:nDimension() > 1 then
            feature = feature[i]
        end 
        local _ , predictions = feature:float():sort(1, true) -- descending
        self.resultTable[idx].pd = predictions[1]
        self.resultTable[idx].ft = torch.Tensor():resizeAs(feature):copy(feature)
    end
end
     


function Trainer:computeScore(output, target, nCrops)
    if nCrops > 1 then
        -- Sum over crops
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
        --:exp()
        :sum(2):squeeze(2)
    end

    if output:nDimension() == 1 then
        output = output:view(1, output:size(1))
    end
    -- Coputes the top1 and top5 error rate
    local batchSize = output:size(1)

    local _ , predictions = output:float():sort(2, true) -- descending

    -- Find which predictions match the target
    local correct = predictions:eq(
    target:long():view(batchSize, 1):expandAs(output))

    -- Top-1 score
    local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

    -- Top-5 score, if there are at least 5 classes
    local len = math.min(5, correct:size(2))
    local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

    return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
    -- if using DataParallelTable. The target is always copied to a CUDA tensor
    if self.opt.visi then
        local sinput = torch.CudaTensor():resize(sample.input[1]:size()):copy(sample.input[1])
        local svisi = torch.CudaTensor():resize(sample.input[2]:size()):copy(sample.input[2])
        self.input = {sinput, svisi}
    else

        self.input = self.input or (self.opt.nGPU == 1
        and torch.CudaTensor()
        or cutorch.createCudaHostTensor())

        self.input:resize(sample.input:size()):copy(sample.input)
    end

    local target = target or (torch.CudaLongTensor and torch.CudaLongTensor() or torch.CudaTensor())
    target:resize(sample.target:size()):copy(sample.target)

    -- use uniform distribution as the desired prob dist,
    -- same fomulation with label smoothing.
    local batch_size = self.opt.visi and sample.input[1]:size(1) or sample.input:size(1)
    local feat_size = self.opt.visi and sample.input[1]:size(2) or sample.input:size(2)
    local uniDist = uniDist or torch.CudaTensor()
    uniDist:resize(batch_size, self.opt.nClasses):fill(1)
    uniDist = uniDist:div(self.opt.nClasses):cuda() 

    if self.opt.siamese ~= 'none' or self.opt.mlpsia then
        self.target = {target, uniDist}
    elseif self.opt.weighted ~= 'none' then
        uniDist:resize(batch_size, feat_size):fill(1)
        uniDist = uniDist:div(feat_size):cuda()
        self.target = {target, uniDist}
    else
        self.target = target
    end
end

function Trainer:learningRate(epoch)
    -- Training schedule
    local decay = math.floor((epoch - 1) / self.opt.decayEpoch)
    return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
