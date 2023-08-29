// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

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

const createBiasAddProgramInfo = (metadata: ProgramMetadata, inputs: readonly TensorView[]): ProgramInfo => {
  const outputShape = inputs[0].dims.slice();

  const channels = inputs[0].dims[2];
  const outputSize = ShapeUtil.size(outputShape);
  const input = inputVariable('input', inputs[0].dataType, inputs[0].dims);
  const bias = inputVariable('bias', inputs[0].dataType, [channels]);
  const residual = inputVariable('residual', inputs[0].dataType, inputs[0].dims);
  const output = outputVariable('output', inputs[0].dataType, outputShape);

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const channels = ${channels}u;
  ${shaderHelper.declareVariables(input, bias, residual, output)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let value = ${input.getByOffset('global_idx')} 
      + ${bias.getByOffset('global_idx % channels')} + ${residual.getByOffset('global_idx')};
    ${output.setByOffset('global_idx', 'value')}
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

  context.compute(createBiasAddProgramInfo(metadata, context.inputs));
};
