--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet and CIFAR-10 datasets
--

local M = {}

function M.create(opt, split)
    local script = paths.dofile(opt.dataset .. '-gen.lua')
    local imageInfo =  script.exec(opt)

    local Dataset = require('datasets/' .. opt.dataset)
    return Dataset(imageInfo, opt, split)
end

return M
