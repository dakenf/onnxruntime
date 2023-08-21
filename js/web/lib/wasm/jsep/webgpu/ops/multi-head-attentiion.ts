import { ComputeContext, GpuDataType } from '../types'
import { TensorView } from '../../tensor'
import { DataType } from '../../../wasm-common'
import { ShaderHelper } from './common'
import { createAttributeWithCacheKey } from '../attribute-with-cache-key'
import { createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata } from './transpose'
import { ShapeUtil } from '../../util'

interface AttentionParameters {
  batchSize: number;
  sequenceLength: number;
  pastSequenceLength: number;
  kvSequenceLength: number
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

interface MultiHeadAttentionAttrs {
  num_heads: number
  is_unidirectional: number
  mask_filter_value: number
  scale: number
  do_rotary: number
}

enum AttentionQkvFormat {
  UNKNOWN,               // enum value not set, or depends on qkv projection implementation details
  Q_K_V_BNSH,            // for non-packed qkv, permuted
  Q_K_V_BSNH,            // for non-packed qkv, not permuted, used by memory efficient attention or MultiHeadAttention
  QKV_BSN3H,             // for TRT fused attention, qkv are packed
  Q_K_V_BNSH_QKV_BS3NH,  // for TRT fused causal attention, data has two formats (qkv is 3BNSH, gemm_buffer is BS3NH)
  Q_KV_BSNH_BSN2H,       // for TRT fused cross attention, kv are packed
  Q_K_V_TNH,             // for memory efficient attention, qkv are not packed, and paddings are removed.
  QKV_TN3H,              // for TRT fused attention, qkv are packed and paddings are removed
}

enum AttentionMaskType {
  MASK_NONE,                  // No mask
  MASK_1D_KEY_SEQ_LEN,        // [batch_size], key sequence length
  MASK_1D_END_START,          // [2 * batch_size] with end positions and start positions
  MASK_1D_KEY_SEQ_LEN_START,  // [3 * batch_size + 2] with [key_len[0], ..., key_len[batch_size - 1], query_start[0],
                              // ..., query_start[batch_size - 1], query_end[batch_size - 1], key_start[0], ...,
                              // key_start[batch_size - 1], key_end[batch_size - 1]]
  MASK_2D_DUMMY,              // dummy mask with shape [1, 1] or [batch_size, 1]. It has same effect as no mask.
  MASK_2D_KEY_PADDING,        // [batch_size, total_sequence_length]
  MASK_3D_ATTENTION,          // [batch_size, sequence_length, total_sequence_length]
  MASK_4D_MEGATRON,           // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
  MASK_UNKNOWN
};

const validateInputs = (inputs: readonly TensorView[], attributes: MultiHeadAttentionAttrs): AttentionParameters => {
  if (!attributes.num_heads) {
    attributes.num_heads = 1;
  }
  const query = inputs[0];
  const key = inputs[1];
  const value = inputs[2];
  // const bias = inputs[3];
  const keyPaddingMask = inputs[4];
  const extraAddQk = inputs[5];
  const pastKey = inputs[6];
  const pastValue = inputs[7];
  if (query.dims.length === 5) {
    throw new Error('Packed QKV of shape (B, L, N, 3, H) not implemented for WebGPU');
  }

  if (key && key.dims.length === 5) {
    throw new Error('Packed KV not implemented for WebGPU');
  }

  if (query.dims.length !== 3 && query.dims.length !== 5) {
    throw new Error('Input query is expected to have 3 or 5 dimensions');
  }

  const dmmhaPacking = false;
  const batchSize = query.dims[0];
  const sequenceLength = query.dims[1];
  const hiddenSize = query.dims.length === 3
    ? (dmmhaPacking ? query.dims[2] / 3 : query.dims[2])
    : attributes.num_heads * query.dims[4];
  let kvSequenceLength = sequenceLength;

  let pastSequenceLength = 0;
  let maxSequenceLength = 0;
  if (pastKey && pastValue) {
    if (pastKey.dims.length !== 4) {
      throw new Error('past_key is expected to have 4 dimensions');
    }

    pastSequenceLength = pastKey.dims[2];
    maxSequenceLength = pastKey.dims[2];
  }

  let qkvFormat = AttentionQkvFormat.Q_K_V_BSNH;
  if (key) {
    if (key.dims.length === 3) {
      qkvFormat = AttentionQkvFormat.Q_K_V_BSNH;
      kvSequenceLength = key.dims[1];
    } else if (key.dims.length === 5) {
      // not implemented
    }
  }

  let maskType: AttentionMaskType = AttentionMaskType.MASK_NONE;
  if (keyPaddingMask) {
    maskType = AttentionMaskType.MASK_UNKNOWN;
    const maskDims = keyPaddingMask.dims;
    if (maskDims.length === 1) {
      if (maskDims[0] === batchSize) {
        maskType = AttentionMaskType.MASK_1D_KEY_SEQ_LEN;
      } else if (maskDims[0] === 3 * batchSize + 2) {
        maskType = AttentionMaskType.MASK_1D_KEY_SEQ_LEN_START
      }
    } else if (maskDims.length === 2 && maskDims[0] === batchSize && maskDims[1] === kvSequenceLength) {
      maskType = AttentionMaskType.MASK_2D_KEY_PADDING;
    }
    if (maskType === AttentionMaskType.MASK_UNKNOWN) {
      throw new Error('Input \'key_padding_mask\' shape shall be (batch_size) or (batch_size, kv_sequence_length)');
    }
  }

  let passPastInKv = false;
  let vHiddenSize = hiddenSize;
  if (value) {
    if (value.dims.length === 3) {
      vHiddenSize = value.dims[2];
    } else {
      vHiddenSize = value.dims[1] * value.dims[3];
      passPastInKv = true;
    }
  }

  let totalSequenceLength = pastSequenceLength + kvSequenceLength;
  let broadcastResPosBias = false;
  if (extraAddQk) {
    if (extraAddQk.dims[0] === 1) {
      broadcastResPosBias = true;
    }
  }

  return {
    batchSize,
    sequenceLength,
    pastSequenceLength,
    kvSequenceLength,
    totalSequenceLength,
    maxSequenceLength,
    inputHiddenSize: 0,
    hiddenSize,
    vHiddenSize,
    headSize: hiddenSize / attributes.num_heads,
    vHeadSize: vHiddenSize / attributes.num_heads,
    numHeads: attributes.num_heads || 1,
    isUnidirectional: false,
    pastPresentShareBuffer: false,
    maskFilterValue: attributes.mask_filter_value,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias,
    passPastInKv,
    qkvFormat,
  }
};


const validateAttentionInputs = (inputs: readonly TensorView[], attributes: MultiHeadAttentionAttrs): AttentionParameters => {
  if (!attributes.num_heads) {
    attributes.num_heads = 1;
  }
  const query = inputs[0];
  const key = inputs[1];
  // const value = inputs[2];
  const bias = inputs[2];
  const keyPaddingMask = inputs[4];
  const extraAddQk = inputs[5];
  const pastKey = inputs[6];
  const pastValue = inputs[7];
  if (query.dims.length === 5) {
    throw new Error('Packed QKV of shape (B, L, N, 3, H) not implemented for WebGPU');
  }

  if (key && key.dims.length === 5) {
    throw new Error('Packed KV not implemented for WebGPU');
  }

  if (query.dims.length !== 3 && query.dims.length !== 5) {
    throw new Error('Input query is expected to have 3 or 5 dimensions');
  }

  // const dmmhaPacking = false;
  const batchSize = query.dims[0];
  const sequenceLength = query.dims[1];
  const hiddenSize = bias.dims[0] / 3;
  // const kHiddenSize = hiddenSize;

  let kvSequenceLength = sequenceLength;

  let pastSequenceLength = 0;
  let maxSequenceLength = 0;
  if (pastKey && pastValue) {
    if (pastKey.dims.length !== 4) {
      throw new Error('past_key is expected to have 4 dimensions');
    }

    pastSequenceLength = pastKey.dims[2];
    maxSequenceLength = pastKey.dims[2];
  }

  let qkvFormat = AttentionQkvFormat.Q_K_V_BSNH;
  if (key) {
    if (key.dims.length === 3) {
      qkvFormat = AttentionQkvFormat.Q_K_V_BSNH;
      kvSequenceLength = key.dims[1];
    } else if (key.dims.length === 5) {
      // not implemented
    }
  }

  let maskType: AttentionMaskType = AttentionMaskType.MASK_NONE;
  if (keyPaddingMask) {
    maskType = AttentionMaskType.MASK_UNKNOWN;
    const maskDims = keyPaddingMask.dims;
    if (maskDims.length === 1) {
      if (maskDims[0] === batchSize) {
        maskType = AttentionMaskType.MASK_1D_KEY_SEQ_LEN;
      } else if (maskDims[0] === 3 * batchSize + 2) {
        maskType = AttentionMaskType.MASK_1D_KEY_SEQ_LEN_START
      }
    } else if (maskDims.length === 2 && maskDims[0] === batchSize && maskDims[1] === kvSequenceLength) {
      maskType = AttentionMaskType.MASK_2D_KEY_PADDING;
    }
    if (maskType === AttentionMaskType.MASK_UNKNOWN) {
      throw new Error('Input \'key_padding_mask\' shape shall be (batch_size) or (batch_size, kv_sequence_length)');
    }
  }

  let passPastInKv = false;
  let vHiddenSize = hiddenSize;

  let totalSequenceLength = pastSequenceLength + kvSequenceLength;
  let broadcastResPosBias = false;
  if (extraAddQk) {
    if (extraAddQk.dims[0] === 1) {
      broadcastResPosBias = true;
    }
  }

  return {
    batchSize,
    sequenceLength,
    pastSequenceLength,
    kvSequenceLength,
    totalSequenceLength,
    maxSequenceLength,
    inputHiddenSize: 0,
    hiddenSize,
    vHiddenSize,
    headSize: hiddenSize / attributes.num_heads,
    vHeadSize: vHiddenSize / attributes.num_heads,
    numHeads: attributes.num_heads || 1,
    isUnidirectional: false,
    pastPresentShareBuffer: false,
    maskFilterValue: attributes.mask_filter_value,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias,
    passPastInKv,
    qkvFormat,
  }
};

export const parseMultiHeadAttentionAttributes = (attributes: MultiHeadAttentionAttrs): MultiHeadAttentionAttrs =>
  createAttributeWithCacheKey({ ...attributes });

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [0, 2, 1, 3]});
const maybeTransposeToBNSHAndAddBias = (context: ComputeContext, batchSize: number, numHeads: number,
  sequenceLength: number, headSize: number, input: TensorView, bias?: TensorView, biasOffset?: number) => {
  // const newDims = [];

  if (bias) {
    if (input.dims.length === 3) {
      input.reshape([batchSize, sequenceLength, numHeads, headSize]);
    }
    return context.compute(
        {
          ...transposeProgramMetadata,
          cacheHint: weightTransposeAttribute.cacheKey,
          get: () => createTransposeProgramInfo(input, weightTransposeAttribute.perm)
        },
        {inputs: [input], outputs: [-1]})[0];
  } else {
    if (sequenceLength === 1) {
      throw new Error('implement AddBiasReshape');
    } else {
      throw new Error('implement AddBiasTranspose');
    }
  }
}

const computeInPlaceSoftmax = (context: ComputeContext, input: TensorView, N: number, D: number) => {
  const dataType = 'f32';

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const dInv = 1 / ${D};
  @group(0) @binding(0) var<storage, read_write> x: array<${dataType}>;

  ${shaderHelper.mainStart()}
    let offset: u32 = global_id.x * ${D};
    
    var threadMax = -3.402823e+38f; // 6.2.4 in wgsl spec
    for (var i: u32 = 0; i < ${D}; i++) {
      threadMax = max(x[offset + i], threadMax);
    }

    for (var i: u32 = 0; i < ${D}; i++) {
      x[offset + i] = exp(x[offset + i] - threadMax);
    }

    var sum = 0.0;
    for (var i: u32 = 0; i < ${D}; i++) {
      sum += x[offset + i];
    }
    
    if (sum == 0) {
      for (var i: u32 = 0; i < ${D}; i++) {
        x[offset + i] = dInv;
      }
    } else {
      for (var i: u32 = 0; i < ${D}; i++) {
        x[offset + i] = x[offset + i] / sum;
      }
    }
  }`;

  context.compute({
    name: 'computeAttentionProbsSoftmax',
    cacheHint: '0',
    inputTypes: [GpuDataType.default],
    outputs: [],
    getShaderSource,
    dispatchGroup: () => ({x: Math.ceil(N / 64 /* workgroup size */)})
  }, { inputs: [input], outputs: []});
}

const computeAttentionProbs = (context: ComputeContext, Q: TensorView, Ki: TensorView, bias: TensorView|undefined,
  parameters: AttentionParameters, attributes: MultiHeadAttentionAttrs) => {
  const probsShape = [
    parameters.batchSize,
    parameters.numHeads,
    parameters.sequenceLength,
    parameters.kvSequenceLength + parameters.pastSequenceLength
  ];
  // TODO: handle mask

  const alpha = attributes.scale === 0 ? 1.0 / Math.sqrt(parameters.headSize) : attributes.scale;
  const gemmSize = parameters.sequenceLength * parameters.totalSequenceLength;
  const unitsOfWork = parameters.batchSize * parameters.numHeads * gemmSize;
  const dataType = 'f32';

  const M = parameters.sequenceLength;
  const N = parameters.totalSequenceLength;
  const K = parameters.headSize;

  console.log('probs', parameters, attributes, M, N, K, probsShape);

  const inputDeclarations = [
    `@group(0) @binding(0) var<storage, read> a: array<${dataType}>;`,
    `@group(0) @binding(1) var<storage, read> b: array<${dataType}>;`,
  ];
  if (bias) {
    inputDeclarations.push(`@group(0) @binding(2) var<storage, read> bias: array<${dataType}>;`)
  }
  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K}u;
  const numHeads: u32 = ${parameters.numHeads};
  const batchSize: u32 = ${parameters.batchSize};
  const gemmSize: u32 = ${gemmSize};
  const alpha = ${dataType}(${alpha});
  const beta = 1.0;

  ${inputDeclarations.join('\n')}
  @group(0) @binding(${inputDeclarations.length}) var<storage, read_write> output: array<${dataType}>;

  ${shaderHelper.mainStart()}
    let batchIndex = global_idx / numHeads / gemmSize;
    let headIndex = global_idx / gemmSize;
    let outputOffset = headIndex * gemmSize;
    let kOffset = ${parameters.kvSequenceLength * parameters.headSize} * headIndex;
    

    let m = global_id.x / batchSize / numHeads / N;
    let n = (global_id.x / batchSize / numHeads) % N;

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      // no trans a + trans b
      value += a[k * M + m] * b[k * N + n];
    }

    value *= alpha;
    // value += beta * output[global_id.x]; // no mask
    ${bias ? 'value += bias[global_id.x]' : ''};
    output[global_id.x] = value;
  }`;

  const inputTypes = [GpuDataType.default, GpuDataType.default];
  const inputs = [Q, Ki];
  if (bias) {
    inputTypes.push(GpuDataType.default);
    inputs.push(bias);
  }

  const probs = context.compute({
    name: 'computeAttentionProbs',
    cacheHint: '0',
    inputTypes,
    outputs: [{dims: probsShape, dataType: Q.dataType, gpuDataType: GpuDataType.default}],
    getShaderSource,
    dispatchGroup: () => ({x: Math.ceil(unitsOfWork / 64 /* workgroup size */)})
  }, { inputs, outputs: [-1]})[0];

  computeInPlaceSoftmax(context, probs, parameters.batchSize * parameters.numHeads * parameters.sequenceLength,
    parameters.totalSequenceLength);

  return probs;
}

const computeVxAttentionScore = (params: AttentionParameters) => {
  const attentionScoreMatMulProgramData = {
    name: 'computeVxAttentionScore',
    inputTypes: [GpuDataType.default, GpuDataType.default],
    cacheHint: '0',
  };

  const outputShape = [params.batchSize, params.numHeads, params.sequenceLength, params.vHeadSize];
  const outputSize = ShapeUtil.size(outputShape);

  const dataType = 'f32';
  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${params.sequenceLength}u;
  const N: u32 = ${params.vHeadSize}u;
  const K: u32 = ${params.totalSequenceLength}u;

  @group(0) @binding(0) var<storage, read> a : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> b : array<${dataType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart()}

    let stack = global_idx / (M * N);
    let mn = global_idx % (M * N);
    let n = global_idx % N;
    let m = mn / N;

    let offsetA = stack * (M * K) + m * K;
    let offsetB = stack * (K * N) + n;

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<K; k++) {
      value += a[offsetA + k] * b[offsetB + k * N];
    }
    output[global_idx] = value;
  }`;
  return {
    ...attentionScoreMatMulProgramData,
    outputs: [{dims: outputShape, dataType: DataType.float, gpuDataType: GpuDataType.default}],
    getShaderSource,
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
  };
}

const applyAttention = (context: ComputeContext, Q: TensorView, K: TensorView, V: TensorView,
  maskIndex: TensorView|undefined, past: TensorView|undefined, pastKey: TensorView|undefined,
  pastValue: TensorView|undefined, relativePositionBias: TensorView|undefined, parameters: AttentionParameters,
  attributes: MultiHeadAttentionAttrs) => {

  const probs = computeAttentionProbs(context, Q, K, relativePositionBias, parameters, attributes);

  const attentionScoreMatMulProgramData = {
    name: 'Transpose',
    inputTypes: [GpuDataType.default, GpuDataType.default],
    cacheHint: JSON.stringify(parameters) + JSON.stringify(attributes),
  };

  const attentionResult = context.compute(
    {
      ...attentionScoreMatMulProgramData,
      cacheHint: JSON.stringify(parameters) + JSON.stringify(attributes),
      get: () => computeVxAttentionScore(parameters)
    },
    {inputs: [probs, V], outputs: [-1]})[0];

  context.compute(
    {
      ...transposeProgramMetadata,
      cacheHint: JSON.stringify(parameters) + JSON.stringify(attributes),
      get: () => createTransposeProgramInfo(attentionResult, weightTransposeAttribute.perm,
        [parameters.batchSize, parameters.sequenceLength, parameters.vHiddenSize])
    },
    {inputs: [attentionResult], outputs:[0]});
}

export const attention = (context: ComputeContext, attributes: MultiHeadAttentionAttrs): void => {
  if (!attributes.num_heads) {
    attributes.num_heads = 1;
  }
  const params = validateAttentionInputs(context.inputs, attributes);
  return applyAttention(context, context.inputs[0], context.inputs[1], context.inputs[2], context.inputs[4],
    undefined, undefined, undefined, context.inputs[5], params, attributes);
}

export const multiHeadAttention = (context: ComputeContext, attributes: MultiHeadAttentionAttrs): void => {
  const params = validateInputs(context.inputs, attributes);

  //const outputShape = [params.batchSize, params.sequenceLength, params.vHiddenSize];
  //const presentKShape = [params.batchSize, attributes.num_heads, params.totalSequenceLength, params.headSize];
  //const presentVShape = [params.batchSize, attributes.num_heads, params.totalSequenceLength, params.vHeadSize];

  const kvBNSH = context.inputs[1] && context.inputs[2] && context.inputs[1].dims.length === 4 &&
    context.inputs[2].dims.length === 4;

  const Q = maybeTransposeToBNSHAndAddBias(context, params.batchSize, params.numHeads, params.sequenceLength,
    params.headSize, context.inputs[0], context.inputs[3], 0);

  if (kvBNSH) {
    return applyAttention(context, Q, context.inputs[1], context.inputs[2], context.inputs[4],
      undefined, undefined, undefined, context.inputs[5], params, attributes);
  }

  const K = maybeTransposeToBNSHAndAddBias(context, params.batchSize, params.numHeads,
    params.kvSequenceLength, params.headSize, context.inputs[1], context.inputs[3], params.hiddenSize);
  const V = maybeTransposeToBNSHAndAddBias(context, params.batchSize, params.numHeads,
    params.kvSequenceLength, params.vHeadSize, context.inputs[2], context.inputs[3], 2 *params.hiddenSize);

  applyAttention(context, Q, K, V, context.inputs[4], undefined,  context.inputs[6],  context.inputs[7],
    context.inputs[5], params, attributes);
};
