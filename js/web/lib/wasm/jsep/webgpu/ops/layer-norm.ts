// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {
  castToF32,
  fillVector,
  getMaxComponents,
  inputVariable,
  outputVariable,
  ShaderHelper,
  sumVector,
  tensorTypeToWsglStorageType,
} from './common';
import { DataType } from '../../../wasm-common';

export interface LayerNormAttributes extends AttributeWithCacheKey {
  axis: number;
  epsilon: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length < 2) {
    throw new Error('layerNorm requires at least 2 inputs.');
  }
};

const createLayerNormProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: LayerNormAttributes, outputCount: number):
        ProgramInfo => {
          const xShape = inputs[0].dims;
          const scale = inputs[1];
          const bias = inputs[2];

          const outputShape = xShape;
          const axis = ShapeUtil.normalizeAxis(attributes.axis, xShape.length);
          const normCount = ShapeUtil.sizeToDimension(xShape, axis);
          const normSize = ShapeUtil.sizeFromDimension(xShape, axis);

          const scaleSize = ShapeUtil.size(scale.dims);
          const biasSize = bias ? ShapeUtil.size(bias.dims) : 0;
          if (scaleSize !== normSize || (bias && biasSize !== normSize)) {
            throw new Error(`Size of X.shape()[axis:] == ${normSize}.
       Size of scale and bias (if provided) must match this.
       Got scale size of ${scaleSize} and bias size of ${biasSize}`);
          }

          const meanInvStdDevDim = [];
          for (let i = 0; i < xShape.length; ++i) {
            if (i < axis) {
              meanInvStdDevDim.push(xShape[i]);
            } else {
              meanInvStdDevDim.push(1);
            }
          }

          // TODO: for some reason it does not work correctly with fp16
          const components = inputs[0].dataType !== DataType.float16 ? getMaxComponents(normSize) : 1;
          const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
          const variables = [
            inputVariable('x', inputs[0].dataType, inputs[0].dims, components),
            inputVariable('scale', scale.dataType, scale.dims, components),
          ];
          if (bias) {
            variables.push(inputVariable('bias', bias.dataType, bias.dims, components));
          }
          variables.push(outputVariable('output', inputs[0].dataType, outputShape, components));

          const hasMeanDataOutput = outputCount > 1;
          const hasInvStdOutput = outputCount > 2;

          if (hasMeanDataOutput) {
            variables.push(outputVariable('meanDataOutput', DataType.float, meanInvStdDevDim));
          }
          if (hasInvStdOutput) {
            variables.push(outputVariable('invStdOutput', DataType.float, meanInvStdDevDim));
          }

          const getShaderSource = (shaderHelper: ShaderHelper) => `
  const normSize: u32 = ${normSize / components};
  const epsilon: f32 = ${attributes.epsilon};

  ${shaderHelper.declareVariables(...variables)}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(normCount)}
    let offset = global_idx * normSize;
    var meanVector = ${fillVector('f32', components)};
    var meanSquareVector = ${fillVector('f32', components)};

    for (var h: u32 = 0u; h < normSize; h++) {
      let value = ${castToF32(dataType, components, 'x[h + offset]')};
      meanVector += value;
      meanSquareVector += value * value;
    }
    let mean = ${sumVector('meanVector', components)} / f32(normSize);
    let meanSquare = sqrt(${sumVector('meanSquareVector', components)} 
      / f32(normSize) - mean * mean + epsilon);

    for (var j: u32 = 0; j < normSize; j++) {
      let f32input = ${castToF32(dataType, components, 'x[j + offset]')};
      let f32scale = ${castToF32(dataType, components, 'scale[j]')};
      output[j + offset] = ${variables[0].type.value}((f32input - mean) / meanSquare * f32scale
        ${bias ? `+ ${castToF32(dataType, components, 'bias[j]')}` : ''}
      );
    }

    ${hasMeanDataOutput ? 'meanDataOutput[global_idx] = mean' : ''};
    ${hasInvStdOutput ? 'invStdOutput[global_idx] = 1 / meanSquare' : ''};
  }`;
          const outputs = [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}];
          if (hasMeanDataOutput) {
            outputs.push({dims: meanInvStdDevDim, dataType: DataType.float, gpuDataType: GpuDataType.default});
          }
          if (hasInvStdOutput) {
            outputs.push({dims: meanInvStdDevDim, dataType: DataType.float, gpuDataType: GpuDataType.default});
          }

          return {
            ...metadata,
            outputs,
            getShaderSource,
            dispatchGroup: () => ({x: Math.ceil(normCount / 64 /* workgroup size */)})
          };
        };

export const parseLayerNormAttributes = (attributes: LayerNormAttributes): LayerNormAttributes =>
    createAttributeWithCacheKey({axis: attributes.axis, epsilon: attributes.epsilon});

export const layerNorm = (context: ComputeContext, attributes: LayerNormAttributes): void => {
  validateInputs(context.inputs);

  const metadata = {
    name: 'LayerNormalization',
    inputTypes: context.inputs.length === 2 ? [GpuDataType.default, GpuDataType.default] :
                                              [GpuDataType.default, GpuDataType.default, GpuDataType.default],
    cacheHint: attributes.cacheKey + context.outputCount.toString(10) + context.inputs.length.toString(10),
  };

  context.compute(createLayerNormProgramInfo(metadata, context.inputs, attributes, context.outputCount));
};
