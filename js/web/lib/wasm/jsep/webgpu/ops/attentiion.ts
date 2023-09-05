// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common'
import {TensorView} from '../../tensor'
import {ShapeUtil} from '../../util'
import {createAttributeWithCacheKey} from '../attribute-with-cache-key'
import {ComputeContext, GpuDataType} from '../types'

import {
  fillVector,
  getMaxComponents,
  inputVariable,
  outputVariable,
  ShaderHelper,
  sumVector,
  tensorTypeToWsglStorageType
} from './common'
import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from './transpose'

export enum AttentionQkvFormat {
  UNKNOWN,               // enum value not set, or depends on qkv projection implementation details
  Q_K_V_BNSH,            // for non-packed qkv, permuted
  Q_K_V_BSNH,            // for non-packed qkv, not permuted, used by memory efficient attention or MultiHeadAttention
  QKV_BSN3H,             // for TRT fused attention, qkv are packed
  Q_K_V_BNSH_QKV_BS3NH,  // for TRT fused causal attention, data has two formats (qkv is 3BNSH, gemm_buffer is BS3NH)
  Q_KV_BSNH_BSN2H,       // for TRT fused cross attention, kv are packed
  Q_K_V_TNH,             // for memory efficient attention, qkv are not packed, and paddings are removed.
  QKV_TN3H,              // for TRT fused attention, qkv are packed and paddings are removed
}

export enum AttentionMaskType {
  MASK_NONE,                  // No mask
  MASK_1D_KEY_SEQ_LEN,        // [batch_size], key sequence length
  MASK_1D_END_START,          // [2 * batch_size] with end positions and start positions
  MASK_1D_KEY_SEQ_LEN_START,  // [3 * batch_size + 2] with [key_len[0], ..., key_len[batch_size - 1], query_start[0],
                              // ..., query_start[batch_size - 1], query_end[batch_size - 1], key_start[0], ...,
                              // key_start[batch_size - 1], key_end[batch_size - 1]]
  MASK_2D_DUMMY,              // dummy mask with shape [1, 1] or [batch_size, 1]. It has same effect as no mask.
  MASK_2D_KEY_PADDING,        // [batch_size, total_sequence_length]
  MASK_3D_ATTENTION,          // [batch_size, sequence_length, total_sequence_length]
  MASK_4D_MEGATRON,  // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
  MASK_UNKNOWN
}
;

export interface AttentionParameters {
  batchSize: number;
  sequenceLength: number;
  pastSequenceLength: number;
  kvSequenceLength: number;
  totalSequenceLength: number;
  maxSequenceLength: number;
  inputHiddenSize: number;
  hiddenSize: number;
  vHiddenSize: number;
  headSize: number;
  vHeadSize: number;
  numHeads: number;
  isUnidirectional: boolean;
  pastPresentShareBuffer: boolean;
  maskFilterValue: number;
  maskType: AttentionMaskType;
  scale: number;
  broadcastResPosBias: boolean;
  passPastInKv: boolean;
  qkvFormat: AttentionQkvFormat;
}

export interface AttentionAttrs {
  numHeads: number;
  isUnidirectional: number;
  maskFilterValue: number;
  scale: number;
  doRotary: number;
  qkvHiddenSizes: number[];
  pastPresentShareBuffer: boolean;
}

const validateAttentionInputs = (inputs: readonly TensorView[], attributes: AttentionAttrs): AttentionParameters => {
  const input = inputs[0];
  const weights = inputs[1];
  const bias = inputs[2];
  const maskIndex = inputs[3];
  const past = inputs[4];
  const relativePositionBias = inputs[5];

  if (past && relativePositionBias) {
    throw new Error('Attention cannot have both past and relative_position_bias');
  }

  if (input.dims.length !== 3) {
    throw new Error('Input "input" must have 3 dimensions');
  }

  const batchSize = input.dims[0];
  const sequenceLength = input.dims[1];
  const inputHiddenSize = input.dims[2];

  if (bias.dims.length !== 1) {
    throw new Error('Input "bias" is expected to have 1 dimensions');
  }

  if (weights.dims.length !== 2) {
    throw new Error('Input "weights" is expected to have 2 dimensions');
  }

  if (weights.dims[0] !== inputHiddenSize) {
    throw new Error('Input 1 dimension 0 should have same length as dimension 2 of input 0');
  }

  if (bias.dims[0] !== weights.dims[1]) {
    throw new Error('Input "bias" dimension 0 should have same length as dimension 1 of input "weights"');
  }

  let qHiddenSize = bias.dims[0] / 3;
  let kHiddenSize = qHiddenSize;
  let vHiddenSize = kHiddenSize;
  if (attributes.qkvHiddenSizes.length > 0) {
    if (attributes.qkvHiddenSizes.length !== 3) {
      throw new Error('qkv_hidden_sizes attribute should have 3 elements');
    }
    for (const sz of attributes.qkvHiddenSizes) {
      if (sz % attributes.numHeads !== 0) {
        throw new Error('qkv_hidden_sizes should be divisible by num_heads');
      }
    }

    qHiddenSize = attributes.qkvHiddenSizes[0];
    kHiddenSize = attributes.qkvHiddenSizes[1];
    vHiddenSize = attributes.qkvHiddenSizes[2];
  }

  const kvSequenceLength = sequenceLength;

  if (qHiddenSize !== kHiddenSize) {
    throw new Error('qkv_hidden_sizes first element should be same as the second');
  }

  if (bias.dims[0] !== qHiddenSize + kHiddenSize + vHiddenSize) {
    throw new Error('Input "bias" dimension 0 should have same length as sum of Q/K/V hidden sizes');
  }

  let pastSequenceLength = 0;
  if (past) {
    if (kHiddenSize !== vHiddenSize) {
      throw new Error('Input "past" expect k_hidden_size == v_hidden_size');
    }
    if (past.dims.length !== 5) {
      throw new Error('Input "past" must have 5 dimensions');
    }
    if (past.dims[0] !== 2) {
      throw new Error('Input "past" first dimension must be 2');
    }
    if (past.dims[1] !== batchSize) {
      throw new Error('Input "past" second dimension must be batch_size');
    }
    if (past.dims[2] !== attributes.numHeads) {
      throw new Error('Input "past" third dimension must be num_heads');
    }
    if (past.dims[4] !== kHiddenSize / attributes.numHeads) {
      throw new Error('Input "past" fifth dimension must be k_hidden_size / num_heads');
    }

    if (!attributes.pastPresentShareBuffer) {
      pastSequenceLength = past.dims[3];
    }
    // TODO: handle past_seq_len
  }

  const totalSequenceLength = kvSequenceLength + pastSequenceLength;
  const maxSequenceLength = -1;

  let maskType = AttentionMaskType.MASK_NONE;
  if (maskIndex) {
    // maskType = AttentionMaskType.MASK_UNKNOWN;
    // TODO: handle mask
    throw new Error('Mask not supported');
  }

  if (past) {
    throw new Error('past is not supported');
  }
  if (relativePositionBias) {
    throw new Error('relativePositionBias is not supported');
  }

  return {
    batchSize,
    sequenceLength,
    pastSequenceLength,
    kvSequenceLength,
    totalSequenceLength,
    maxSequenceLength,
    inputHiddenSize,
    hiddenSize: qHiddenSize,
    vHiddenSize,
    headSize: Math.floor(qHiddenSize / attributes.numHeads),
    vHeadSize: Math.floor(vHiddenSize / attributes.numHeads),
    numHeads: attributes.numHeads,
    isUnidirectional: false,
    pastPresentShareBuffer: false,
    maskFilterValue: attributes.maskFilterValue,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias: false,
    passPastInKv: false,
    qkvFormat: AttentionQkvFormat.Q_K_V_BNSH,
  };
};

export const parseAttentionAttributes = (attributes: AttentionAttrs): AttentionAttrs =>
  createAttributeWithCacheKey({...attributes});

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [0, 2, 1, 3]});

export const computeInPlaceSoftmax = (context: ComputeContext, input: TensorView, N: number, D: number) => {
  const components = getMaxComponents(D);
  const inputHelper = outputVariable('x', input.dataType, input.dims, components);

  let threadMaxValue = 'threadMaxVector';
  if (components === 2) {
    threadMaxValue = 'max(threadMaxVector.x, threadMaxVector.y)';
  } else if (components === 4) {
    threadMaxValue = 'max(max(threadMaxVector.x, threadMaxVector.y), max(threadMaxVector.z, threadMaxVector.w))';
  }
  const dataType = tensorTypeToWsglStorageType(input.dataType);
  const threadMaxMinValue = dataType === 'f32' ? '-3.402823e+38f' : '-65504.0h';
  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const dInv: ${dataType} = 1 / ${D};
  const dComp = ${D / components};
  ${shaderHelper.declareVariables(inputHelper)}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(N)}
    let offset: u32 = global_idx * dComp;

    var threadMaxVector = ${fillVector(dataType, components, threadMaxMinValue)}; // 6.2.4 in wgsl spec
    for (var i: u32 = 0; i < dComp; i++) {
      threadMaxVector = max(x[offset + i], threadMaxVector);
    }
    let threadMax: ${dataType} = ${threadMaxValue};

    for (var i: u32 = 0; i < dComp; i++) {
      let val = x[offset + i] - threadMax;
      x[offset + i] = exp(val);
    }

    var sumVector = ${fillVector(dataType, components, '0')};
    for (var i: u32 = 0; i < dComp; i++) {
      sumVector += x[offset + i];
    }
    let sum = ${sumVector('sumVector', components)};

    if (sum == 0) {
      for (var i: u32 = 0; i < dComp; i++) {
        x[offset + i] = ${fillVector(dataType, components, 'dInv')};
      }
    } else {
      for (var i: u32 = 0; i < dComp; i++) {
        x[offset + i] = x[offset + i] / sum;
      }
    }
  }`;

  context.compute(
    {
      name: 'computeAttentionProbsSoftmax',
      cacheHint: '0',
      inputTypes: [GpuDataType.default],
      outputs: [],
      getShaderSource,
      dispatchGroup: () => ({x: Math.ceil(N / 64 /* workgroup size */)})
    },
    {inputs: [input], outputs: []});
};

const computeAttentionProbs =
  (context: ComputeContext, q: TensorView, key: TensorView, bias: TensorView|undefined,
    parameters: AttentionParameters, attributes: AttentionAttrs) => {
    const probsShape = [
      parameters.batchSize, parameters.numHeads, parameters.sequenceLength,
      parameters.kvSequenceLength + parameters.pastSequenceLength
    ];
    // TODO: handle mask

    const alpha = attributes.scale === 0 ? 1.0 / Math.sqrt(parameters.headSize) : attributes.scale;
    const gemmSize = parameters.sequenceLength * parameters.totalSequenceLength;

    const dataType = tensorTypeToWsglStorageType(q.dataType);

    const components = getMaxComponents(parameters.headSize);
    const qInput = inputVariable('q', q.dataType, q.dims, components);
    const kInput = inputVariable('key', key.dataType, key.dims, components);
    const output = outputVariable('output', q.dataType, probsShape);

    const vectorizedHeadSize = parameters.headSize / components;
    const M = parameters.sequenceLength;
    const N = parameters.totalSequenceLength;
    const K = vectorizedHeadSize;

    const unitsOfWork = ShapeUtil.size(probsShape);

    const inputs = [q, key];
    const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K}u;
  const numHeads: u32 = ${parameters.numHeads};
  const batchSize: u32 = ${parameters.batchSize};
  const gemmSize: u32 = ${gemmSize};
  const alpha = ${dataType}(${alpha});
  const beta: ${dataType} = 1.0;

  ${shaderHelper.declareVariables(qInput, kInput, output)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(unitsOfWork)}
    let idxWoGemmSize = global_idx / gemmSize;
    let batchIndex = idxWoGemmSize / numHeads;
    let headIndex = idxWoGemmSize % numHeads;
    let inputOffset = ${parameters.sequenceLength * vectorizedHeadSize} * idxWoGemmSize;
    let kOffset = ${parameters.kvSequenceLength * vectorizedHeadSize} * idxWoGemmSize;

    let gemmOffset = global_idx % gemmSize;
    let m = gemmOffset / N;
    let n = gemmOffset % N;

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      // no trans a + trans b
      value += ${components > 1 ? 'dot(q[m * K + k + inputOffset], key[n * K + k + kOffset])' 
      : 'q[m * K + k + inputOffset] * key[n * K + k + kOffset]'};
    }
    // value += beta * output[global_id.x]; // no mask
    output[global_idx] = value * alpha;
  }`;

    const inputTypes = inputs.map(_ => GpuDataType.default);

    const probs = context.compute(
      {
        name: 'computeAttentionProbs',
        cacheHint: JSON.stringify(parameters),
        inputTypes,
        outputs: [{dims: probsShape, dataType: q.dataType, gpuDataType: GpuDataType.default}],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(unitsOfWork / 64 /* workgroup size */)})
      },
      {inputs, outputs: [-1]})[0];

    computeInPlaceSoftmax(
      context, probs, parameters.batchSize * parameters.numHeads * parameters.sequenceLength,
      parameters.totalSequenceLength);

    return probs;
  };

const computeVxAttentionScore = (context: ComputeContext, probs: TensorView, v: TensorView, params: AttentionParameters) => {
  const outputShape = [params.batchSize, params.numHeads, params.sequenceLength, params.vHeadSize];
  const outputSize = ShapeUtil.size(outputShape);

  const probsHelper = inputVariable('probs', probs.dataType, probs.dims);
  const vHelper = inputVariable('v', v.dataType, v.dims);
  const output = outputVariable('output', probs.dataType, outputShape);

  const dataType = tensorTypeToWsglStorageType(probs.dataType);
  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${params.sequenceLength}u;
  const N: u32 = ${params.vHeadSize}u;
  const K: u32 = ${params.totalSequenceLength}u;
  const numHeads: u32 = ${params.numHeads}u;

  ${shaderHelper.declareVariables(probsHelper, vHelper, output)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let n = global_idx % N;
    let m = (global_idx / N) % M;
    let stack = global_idx / (M * N);
    let batchIndex = stack / numHeads;
    let headIndex = stack % numHeads;

    let offsetA = stack * (M * K) + m * K;
    let offsetB = stack * (K * N) + n;

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<K; k++) {
      value += probs[offsetA + k] * v[offsetB + k * N];
    }
    output[global_idx] = value;
  }`;

  return context.compute(
    {
      name: 'AttentionScore',
      inputTypes: [GpuDataType.default, GpuDataType.default],
      cacheHint: JSON.stringify(params),
      outputs: [{dims: outputShape, dataType: DataType.float, gpuDataType: GpuDataType.default}],
      getShaderSource,
      dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
    },
    {inputs: [probs, v], outputs: [-1]})[0];
};

export const applyAttention =
  (context: ComputeContext, q: TensorView, k: TensorView, v: TensorView, maskIndex: TensorView|undefined,
    past: TensorView|undefined, pastKey: TensorView|undefined, pastValue: TensorView|undefined,
    relativePositionBias: TensorView|undefined, parameters: AttentionParameters, attributes: AttentionAttrs) => {
    const probs = computeAttentionProbs(context, q, k, relativePositionBias, parameters, attributes);

    const attentionResult = computeVxAttentionScore(context, probs, v, parameters);

    context.compute(
      {
        ...transposeProgramMetadata,
        cacheHint: JSON.stringify(parameters) + JSON.stringify(attributes),
        get: () => createTransposeProgramInfo(
          attentionResult, weightTransposeAttribute.perm,
          [parameters.batchSize, parameters.sequenceLength, parameters.vHiddenSize])
      },
      {inputs: [attentionResult], outputs: [0]});
  };

const prepare = (context: ComputeContext, parameters: AttentionParameters, attributes: AttentionAttrs) => {
  const outputShape = [
    parameters.batchSize,
    parameters.numHeads,
    parameters.sequenceLength,
    parameters.headSize,
  ];
  // TODO: handle mask

  // const alpha = attributes.scale === 0 ? 1.0 / Math.sqrt(parameters.headSize) : attributes.scale;
  const gemmSize = parameters.sequenceLength * parameters.hiddenSize;
  const unitsOfWork = gemmSize * parameters.batchSize * parameters.numHeads * 3;
  const dataType = tensorTypeToWsglStorageType(context.inputs[0].dataType);

  const M = parameters.sequenceLength;
  const K = parameters.inputHiddenSize;

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const K: u32 = ${K}u;
  const numHeads: u32 = ${parameters.numHeads};
  const headSizes = array<u32, 3>(${parameters.headSize}, ${parameters.headSize}, ${parameters.vHeadSize});
  const ldb = ${parameters.hiddenSize + parameters.hiddenSize + parameters.vHiddenSize}u;

  @group(0) @binding(0) var<storage, read> input: array<${dataType}>;
  @group(0) @binding(1) var<storage, read> weight: array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias: array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> outputQ: array<${dataType}>;
  @group(0) @binding(4) var<storage, read_write> outputK: array<${dataType}>;
  @group(0) @binding(5) var<storage, read_write> outputV: array<${dataType}>;

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(unitsOfWork)}
    let qkvIndex = global_idx % 3;
    let globalIdxDiv3 = global_idx / 3;
    let N: u32 = headSizes[qkvIndex];
    let gemmSize = M * N;
    let idxWoGemmSize = globalIdxDiv3 / gemmSize;
    let batchIndex = idxWoGemmSize / numHeads;
    let headIndex = idxWoGemmSize % numHeads;

    let inputOffset = batchIndex * ${parameters.sequenceLength * parameters.inputHiddenSize};
    let biasOffset = qkvIndex * ${parameters.hiddenSize} + headIndex * headSizes[qkvIndex];

    let gemmOffset = globalIdxDiv3 % gemmSize;
    let m = gemmOffset / N;
    let n = gemmOffset % N;

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      // no trans
      value += input[m * K + k + inputOffset] * weight[k * ldb + biasOffset + n];
    }

    value += bias[gemmOffset % headSizes[qkvIndex] + biasOffset];
    if (qkvIndex == 0) {
      outputQ[globalIdxDiv3] = value;
    } else if (qkvIndex == 1) {
      outputK[globalIdxDiv3] = value;
    } else if (qkvIndex == 2) {
      outputV[globalIdxDiv3] = value;
    }
  }`;

  const inputTypes = [GpuDataType.default, GpuDataType.default, GpuDataType.default];
  const inputs = [context.inputs[0], context.inputs[1], context.inputs[2]];

  return context.compute(
    {
      name: 'computeAttentionPrepare',
      cacheHint: JSON.stringify(parameters),
      inputTypes,
      outputs: [
        {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
        {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
        {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
      ],
      getShaderSource,
      dispatchGroup: () => ({x: Math.ceil(unitsOfWork / 64 /* workgroup size */)})
    },
    {inputs, outputs: [-1, -1, -1]});
};

export const attention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateAttentionInputs(context.inputs, attributes);

  const [q, k, v] = prepare(context, params, attributes);

  return applyAttention(
    context, q, k, v, context.inputs[4], undefined, undefined, undefined, context.inputs[5], params, attributes);
};
