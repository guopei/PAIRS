--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'torch'
require 'paths'
require 'optim'
require 'nn'
--require 'tcbp'
require 'dpnn'
require 'os'
require 'gnuplot'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
cutorch.setDevice(opt.GPU)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

--if opt.unbalance then
--print(trainLoader.weight)
--    criterion = nn.CrossEntropyCriterion(trainLoader.weight:cmul(torch.Tensor({1,1,0.01}))):cuda()
--criterion = nn.CrossEntropyCriterion(trainLoader.weight:mul(opt.batchSize)):cuda()
--end

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
    print('testing val set...')
    local top1Err, top5Err = trainer:test(0, valLoader)
    print(string.format(' * Test set results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
    print('testing train set...')
    local top1Err, top5Err = trainer:test(0, trainLoader)
    print(string.format(' * Train set results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
    return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do
    -- Train for a single epoch
    local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

    -- Run model on validation set
    if epoch % opt.testIter == 0 then
        local testTop1, testTop5, testLoss = trainer:test(epoch, valLoader)

        if opt.reweight ~= 'none' and false then
            local save_name = '../ipynb/t7s/sparse-gate-stat.t7'
            local result_table = torch.load(save_name)
            result_table[opt.maxkpool].accuracy = testTop1
            torch.save(save_name, result_table)
        end

        trainer.trainLogger:add{
            ['train accu'] = trainTop1,
            ['train loss'] = trainLoss,
            ['test accu'] = testTop1,
            ['test loss'] = testLoss
        }

        local bestModel = false
        if testTop1 < bestTop1 then
            bestModel = true
            bestTop1 = testTop1
            bestTop5 = testTop5
            print(' * Best model ', testTop1, testTop5)
        end

        checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
    end
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
