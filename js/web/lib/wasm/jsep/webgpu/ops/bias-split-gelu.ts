// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType} from './common';
import {erfImpl} from './unary-op';
import {ShapeUtil} from "../../util";

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('inputs should be float type');
  }

  if (inputs[0].dims.length !== 3) {
    throw new Error('input should have 3 dimensions');
  }

  if (![2560, 5120, 10240].includes(inputs[0].dims[2])) {
    throw new Error('hidden state should be 2560, 5120 or 10240');
  }

  if (inputs[1].dims.length !== 1) {
    throw new Error('bias is expected to have 1 dimensions');
  }

  if (inputs[0].dims[2] !== inputs[1].dims[0]) {
    throw new Error('last dimension of input and bias are not the same');
  }
};

const createBiasSplitGeluProgramInfo = (metadata: ProgramMetadata, inputs: readonly TensorView[]): ProgramInfo => {
  const outputShape = inputs[0].dims.slice();
  outputShape[2] = outputShape[2] / 2;

  const channels = inputs[0].dims[2];
  const outputSize = ShapeUtil.size(outputShape);

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M_SQRT2 = sqrt(2.0);
  const channels = ${channels}u;
  const rightBiasOffset = ${outputShape[2] / 8}u;

  @group(0) @binding(0) var<storage, read> input : array<mat4x2f>;
  @group(0) @binding(1) var<storage, read> bias : array<vec4f>;
  @group(0) @binding(1) var<storage, read> output : array<vec4f>;

  ${erfImpl('f32')}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let biasIdx = global_idx % channels;
    let itemIndex = global_idx;
    let valueLeft = input[global_idx][0] + bias[biasIdx];
    let valueRight = input[global_idx][1] + bias[biasIdx + rightBiasOffset];
    let geluRight = valueRight * 0.5 * (erf_vf32(valueRight / M_SQRT2) + 1);

    output[global_idx] = valueLeft * geluRight;
  }`;

  return {
    ...metadata,
    outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
    getShaderSource,
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
  };
};

export const biasSplitGelu = (context: ComputeContext): void => {
  validateInputs(context.inputs);

  const metadata = {
    name: 'BiasSplitGelu',
    inputTypes: [GpuDataType.default, GpuDataType.default],
  };

  context.compute(createBiasSplitGeluProgramInfo(metadata, context.inputs));
};
