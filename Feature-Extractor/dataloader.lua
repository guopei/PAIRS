--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

require 'nn'
require 'cudnn'
require 'cunn'
local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
    -- The train and val loader
    local loaders = {}

    for i, split in ipairs{'train', 'val'} do
        local dataset = datasets.create(opt, split)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end

    return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    local manualSeed = opt.manualSeed
    local function init()
        require('datasets/' .. opt.dataset)
    end
    local function main(idx)
        if manualSeed ~= 0 then
            torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        _G.dataset = dataset
        _G.preprocess = dataset:preprocess()
        _G.dsname = opt.dataset
        _G.wingStack = opt.wingStack

        return dataset:size()
    end

    local threads, sizes = Threads(opt.nThreads, init, main)
    self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
    self.threads = threads
    self.__size = sizes[1][1]
    self.batchSize = math.floor(opt.batchSize / self.nCrops)
    self.weight = dataset.weight 
    self.split = split
end

function DataLoader:size()
    return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
    local threads = self.threads
    local size, batchSize = self.__size, self.batchSize
    local perm = torch.randperm(size)

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
            --print(idx, math.min(batchSize, size - idx + 1))
            threads:addjob(
            function(indices, nCrops)
                local sz = indices:size(1)
                local batch, imageSize
                local name = {}
                local target = torch.IntTensor(sz)
                local input = {}
                local visi
                --local weight = torch.Tensor(sz):zero()
                for i, idx in ipairs(indices:totable()) do
                    local sample = _G.dataset:get(idx)
                    if _G.dsname == 'imagenet' then
                        input = sample.input
                    elseif _G.dsname == 'hdf5' or _G.dsname == 'mvgg' then
                        -- hard coded
                        input = torch.zeros(3*_G.wingStack, 224, 224)
                        for ch = 1, 3*_G.wingStack, 3 do
                            input:narrow(1,ch,3):copy(_G.preprocess(sample.input:narrow(1,ch,3)))
                        end
                    elseif _G.dsname == 'patches' then
                        -- hard coded
                        input = torch.zeros(3*_G.wingStack, 64, 128)
                        for ch = 1, 3*_G.wingStack, 3 do
                            input:narrow(1,ch,3):copy(_G.preprocess(sample.input:narrow(1,ch,3)))
                        end
                    else
                        input = _G.preprocess(sample.input)
                    end

                    --if _G.dsname == 'sample' then
                    --    weight[i]:copy(sample.weight) 
                    --end

                    if not batch then
                        imageSize = input:size():totable()
                        if nCrops > 1 then table.remove(imageSize, 1) end
                        batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                        visi = torch.FloatTensor(sz*nCrops, 15)
                    end
                    batch[i]:copy(input)
                    target[i] = sample.target
                    name[i] = sample.name
                    if _G.dsname == 'visi' then
                        visi[i] = (sample.visi)
                    end
                end
                collectgarbage()

                local feats = (_G.dsname == 'imagenet') and 
                batch:view(table.unpack(imageSize)) or 
                batch:view(sz * nCrops, table.unpack(imageSize))
                return {
                    input = (_G.dsname == 'visi') and {feats, visi} or feats,
                    target = target,
                    --weight = weight,
                    name = name
                }
            end,
            function(_sample_)
                sample = _sample_
            end,
            indices,
            self.nCrops
            )
            idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader
