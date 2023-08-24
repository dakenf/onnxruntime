// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {ShaderHelper, tensorTypeToWsglStorageType} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('inputs should be float type');
  }

  if (inputs[0].dims.length !== 3) {
    throw new Error('input should have 3 dimensions');
  }

  if (![320, 640, 1280].includes(inputs[0].dims[2])) {
    throw new Error('number of channels should be 320, 640 or 1280');
  }

  if (inputs[1].dims.length !== 1) {
    throw new Error('bias is expected to have 1 dimensions');
  }

  if (inputs[0].dims[2] !== inputs[1].dims[0]) {
    throw new Error('last dimension of input and bias are not the same');
  }
};

const createBiasSplitGeluProgramInfo = (metadata: ProgramMetadata, inputs: readonly TensorView[]): ProgramInfo => {
  const input = inputs[0];
  const outputShape = input.dims.slice();

  const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
  const threadsPerBlock = 64;
  const channels = input.dims[2];
  const blockSize = channels / threadsPerBlock;
  const outputSize = ShapeUtil.size(outputShape) * threadsPerBlock;

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const TPB = ${threadsPerBlock}u;
  @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> residual : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let blockIdx = global_idx / ${channels};
    let threadIdx = global_idx % TPB;

    var baseOffset = blockIdx * ${channels} + threadIdx;
    var biasOffset = threadIdx;

    for (var h: u32 = 0u; h < ${blockSize}; h++) {
      output[baseOffset] = input[baseOffset] + bias[biasOffset] + residual[baseOffset];
      baseOffset += TPB;
      biasOffset += TPB;
    }
  }`;

  return {
    ...metadata,
    outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
    getShaderSource,
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
  };
};

export const biasAdd = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  const inputTypes = Array(context.inputs.length).fill(GpuDataType.default);
  const metadata = {
    name: 'BiasAdd',
    inputTypes,
  };

  context.compute(createBiasSplitGeluProgramInfo(metadata, context.inputs));
};
