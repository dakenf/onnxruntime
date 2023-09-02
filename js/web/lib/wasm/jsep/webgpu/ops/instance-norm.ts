// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {
  ShaderHelper,
  inputVariable,
  tensorTypeToWsglStorageType,
  outputVariable,
  getMaxComponents,
  fillVector
} from './common'

export interface InstanceNormAttributes extends AttributeWithCacheKey {
  epsilon: number;
  format: 'NHWC'|'NCHW';
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 3) {
    throw new Error('instanceNorm requires 3 inputs.');
  }

  if (inputs[0].dataType !== DataType.float || inputs[1].dataType !== DataType.float) {
    throw new Error('inputs should be float type');
  }
};

const createInstanceNormProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: InstanceNormAttributes): ProgramInfo => {
      const xShape = inputs[0].dims;
      const outputShape = xShape;
      const outputSize = ShapeUtil.size(outputShape);
      const axis = 2;
      const normCount = xShape[0] * xShape[1];
      const normSize = ShapeUtil.sizeFromDimension(xShape, axis);
      const C = xShape[1];

      const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const C: u32 = ${C};
  const normSize: u32 = ${normSize};
  const normSizeTyped: ${dataType} = ${normSize};
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart()}
    let offset = global_idx * normSize;
    if (offset >= ${outputSize}) { return; }
    var mean: ${dataType} = 0;

    for (var h: u32 = 0u; h < normSize; h++) {
        mean = mean + x[h + offset];
    }
    mean = mean / normSizeTyped;

    var squaredNorm: ${dataType} = 0;
    for (var h: u32 = 0u; h < normSize; h++) {
        let deviation: f32 = x[h + offset] - mean;
        squaredNorm = squaredNorm + deviation * deviation;
    }
    let invStdDev = 1 / sqrt(squaredNorm / normSizeTyped + epsilon);
    let channelScale = invStdDev * scale[global_idx % C];
    let channelShift = bias[global_idx % C] - mean * channelScale;
    for (var j: u32 = 0; j < normSize; j++) {
        output[j + offset] = x[j + offset] * channelScale + channelShift;
    }
  }`;
      return {
        ...metadata,
        outputs: [
          {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
        ],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(normCount / 64 /* workgroup size */)})
      };
    };

const computeMean = (context: ComputeContext, input: TensorView, scale: TensorView, bias: TensorView, n: number, h: number, c: number, epsilon: number) => {
  const components = getMaxComponents(c);
  const inputHelper = inputVariable('input', input.dataType, input.dims, components);
  const scaleHelper = inputVariable('scale', scale.dataType, scale.dims, components);
  const biasHelper = inputVariable('bias', bias.dataType, bias.dims, components);

  const unitsOfWork = n * c / components;

  // we will store channel scale and channel shift in [2, components] matrix
  // or in vec2 when components == 1
  const outputType = components === 1 ? 'array<vec2<f32>>' : `array<mat2x${components}f>`;
  const setOutputValue = components === 1
    ? 'vec2f(channelScale, channelShift)'
    : `mat2x${components}<f32>(channelScale, channelShift)`;

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const H: u32 = ${h};
  const C: u32 = ${c / components};
  const imageSize: u32 = ${h * c / components};
  const epsilon: f32 = ${epsilon};

  ${shaderHelper.declareVariables(inputHelper, scaleHelper, biasHelper)}
  @group(0) @binding(3) var<storage, read_write> output : ${outputType};
  // @group(0) @binding(4) var<storage, read_write> output2 : array<f32>;

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(unitsOfWork)}
    let currentImageNumber = global_idx / C;
    let currentChannelNumber = global_idx % C;

    let offset = currentImageNumber * imageSize;
    var mean: ${inputHelper.type.storage} = ${fillVector(components)};
    for (var i: u32 = 0u; i < H; i++) {
        mean = mean + input[offset + i * C + currentChannelNumber];
    }
    mean = mean / f32(H);

    var squaredNorm: ${inputHelper.type.storage} = ${fillVector(components)};
    for (var i: u32 = 0u; i < H; i++) {
        let deviation = input[offset + i * C + currentChannelNumber] - mean;
        squaredNorm = squaredNorm + deviation * deviation;
    }
    let invStdDev = 1 / sqrt(squaredNorm / f32(H) + epsilon);
    let channelScale = invStdDev * scale[currentChannelNumber];
    let channelShift = bias[currentChannelNumber] - mean * channelScale;

    output[global_idx] = ${setOutputValue};
  }`;

  return context.compute(
    {
      name: 'InstanceNormComputeMean',
      inputTypes: [GpuDataType.default, GpuDataType.default, GpuDataType.default],
      cacheHint: JSON.stringify({ components, n, h, c, epsilon }),
      outputs: [
        {dims: [n, c, 2], dataType: DataType.float, gpuDataType: GpuDataType.default},
        // {dims: [h * c], dataType: DataType.float, gpuDataType: GpuDataType.default},
      ],
      getShaderSource,
      dispatchGroup: () => ({x: Math.ceil(unitsOfWork / 64 /* workgroup size */)})
    },
    {inputs: [input, scale, bias], outputs: [-1]})[0];
};

const createInstanceNormNHWCProgramInfo =
    (context: ComputeContext, metadata: ProgramMetadata, inputs: readonly TensorView[],
      attributes: InstanceNormAttributes) => {
      const xShape = inputs[0].dims;
      const outputShape = xShape;
      const outputSize = ShapeUtil.size(outputShape);
      const N = xShape[0];
      const C = xShape[xShape.length - 1];
      const H = ShapeUtil.sizeFromDimension(xShape, 1) / C;

      const components = getMaxComponents(C);
      const inputHelper = inputVariable('input', inputs[0].dataType, inputs[0].dims, components);
      const outputHelper = outputVariable('output', inputs[0].dataType, outputShape, components);

      const scaleType = components === 1 ? 'vec2<f32>' : `mat2x${components}f`;
      // first compute mean
      const channelScaleShift = computeMean(context, inputs[0], inputs[1], inputs[2], N, H, C, attributes.epsilon);

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const H: u32 = ${H};
  const C: u32 = ${C / components};

  @group(0) @binding(0) var<storage, read> input : array<${inputHelper.type.storage}>;
  @group(0) @binding(1) var<storage, read> scaleInput : array<${scaleType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${outputHelper.type.storage}>;

  ${shaderHelper.mainStart()}
    let currentImageNumber = global_idx / (C * H);
    let currentChannelNumber = global_idx % C;

    let scaleOffset = currentImageNumber * C + currentChannelNumber;
    let scale = scaleInput[scaleOffset];
    output[global_idx] = input[global_idx] * scale[0] + scale[1];
  }`;
      context.compute({
        ...metadata,
        inputTypes: [GpuDataType.default, GpuDataType.default],
        outputs: [
          {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
        ],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      },
      {
        inputs: [inputs[0], channelScaleShift]
      });
    };

export const parseInstanceNormAttributes = (attributes: InstanceNormAttributes): InstanceNormAttributes =>
    createAttributeWithCacheKey({epsilon: attributes.epsilon, format: attributes.format});

export const instanceNorm = (context: ComputeContext, attributes: InstanceNormAttributes): void => {
  validateInputs(context.inputs);

  const metadata = {
    name: 'InstanceNormalization',
    inputTypes: [GpuDataType.default, GpuDataType.default, GpuDataType.default],
    cacheHint: attributes.cacheKey,
  };

  if (attributes.format === 'NHWC') {
    createInstanceNormNHWCProgramInfo(context, metadata, context.inputs, attributes);
  } else {
    context.compute(createInstanceNormProgramInfo(metadata, context.inputs, attributes));
  }
};
