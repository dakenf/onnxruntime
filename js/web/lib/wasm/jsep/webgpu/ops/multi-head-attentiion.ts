import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType} from '../types';

import {applyAttention, AttentionAttrs, AttentionMaskType, AttentionParameters, AttentionQkvFormat, computeInPlaceSoftmax,} from './attentiion';
import { inputVariable, outputVariable, ShaderHelper, sumVector } from './common'
import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from './transpose';

const validateInputs = (inputs: readonly TensorView[], attributes: AttentionAttrs): AttentionParameters => {
  const query = inputs[0];
  const key = inputs[1];
  const value = inputs[2];
  const bias = inputs[3];
  const keyPaddingMask = inputs[4];
  const relativePositionBias = inputs[5];
  const pastKey = inputs[6];
  const pastValue = inputs[7];

  // Abbreviation and Meanings:
  //   B:    batch_size
  //   S:    sequence_length (input sequence length of query)
  //   P:    past_sequence_length (past sequence length of key or value)
  //   L:    kv_sequence_length (input sequence length of key or value)
  //   M:    max_sequence_length
  //   T:    total_sequence_length = past_sequence_length + kv_sequence_length
  //   N:    num_heads
  //   H:    head size for Q and K, aka q_head_size or k_head_size or qk_head_size
  //   H_v:  v_head_size
  //   D_i:  input hidden size
  //   D:    hidden size for Q and K (D = N * H), aka q_hidden_size or k_hidden_size or qk_hidden_size
  //   D_v:  v_hidden_size = num_heads * v_head_size

  //     key_padding_mask (K/V)     : (B) or (2*B + 1) or (B, L) or None
  //     relative_position_bias     : (B, 1, S, L)
  //     past_key                   : (B, N, S*, H)
  //     past_value                 : (B, N, S*, H)
  // When no packing for q/k/v:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D) or (B, N, S*, H)
  //     value            (V)       : (B, L, D_v) or (B, N, S*, H)
  //     bias             (Q/K/V)   : (D + D + D_v)
  // When packed kv is used:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, N, 2, H)
  //     value            (V)       : None
  //     bias             (Q/K/V)   : None
  // When packed qkv is used:
  //     query            (Q)       : (B, L, N, 3, H) or (B, S, 3*D)
  //     key              (K)       : None
  //     value            (V)       : None
  //     bias             (Q/K/V)   : None or (D + D + D_v)

  if (query.dims.length !== 3 && query.dims.length !== 5) {
    throw new Error('Input query is expected to have 3 or 5 dimensions');
  }

  const dmmhaPacking = false;
  const batchSize = query.dims[0];
  const sequenceLength = query.dims[1];
  const hiddenSize = query.dims.length === 3 ? (dmmhaPacking ? query.dims[2] / 3 : query.dims[2]) :
                                               attributes.numHeads * query.dims[4];
  let kvSequenceLength = sequenceLength;

  let pastSequenceLength = 0;
  let maxSequenceLength = 0;
  const headSize = Math.floor(hiddenSize / attributes.numHeads);
  if (pastKey && pastValue) {
    if (pastKey.dims.length !== 4) {
      throw new Error('Input \'past_key\' is expected to have 4 dimensions');
    }
    if (pastValue.dims.length !== 4) {
      throw new Error('Input \'past_value\' is expected to have 4 dimensions')
    }
    pastSequenceLength = pastKey.dims[2];
    maxSequenceLength = pastKey.dims[2];
  } else if (pastKey || pastValue) {
    throw new Error('Input \'past_key\' and \'past_value\' shall be both present or both absent')
  }

  let qkvFormat: AttentionQkvFormat;
  if (key) {
    if (query.dims.length !== 3) {
      throw new Error('Input \'query\' is expected to have 3 dimensions when key is given');
    }
    if (key.dims.length < 3 || key.dims.length > 5) {
      throw new Error('Input \'key\' is expected to have 3, 4, or 5 dimensions');
    }
    if (query.dims[0] !== key.dims[0]) {
      throw new Error('Input \'query\' and \'key\' shall have same dim 0 (batch size)');
    }

    if (key.dims.length === 3) {
      if (key.dims[2] !== query.dims[2]) {
        throw new Error('Input \'query\' and \'key\' shall have same dim 2 (hidden_size)');
      }
      qkvFormat = AttentionQkvFormat.Q_K_V_BSNH;
      kvSequenceLength = key.dims[1];
    } else if (key.dims.length === 5) {
      if (key.dims[2] !== attributes.numHeads || key.dims[3] !== 2 || key.dims[4] !== headSize) {
        throw new Error('Expect \'key\' shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv');
      }
      if (value) {
        throw new Error('Expect \'value\' be none when \'key\' has packed kv format.');
      }
      qkvFormat = AttentionQkvFormat.Q_KV_BSNH_BSN2H;
      kvSequenceLength = key.dims[1];
    } else {  // key_dims.size() == 4 (cross-attention with past_key)
      if (key.dims[1] !== attributes.numHeads || key.dims[3] !== headSize) {
        throw new Error('Expect \'key\' shape (batch_size, num_heads, kv_sequence_length, head_size) for past_key');
      }

      qkvFormat = AttentionQkvFormat.UNKNOWN;
      kvSequenceLength = key.dims[2];
    }
  } else {  // packed QKV
    if (query.dims.length !== 3 && query.dims.length !== 5) {
      throw new Error('Input \'query\' is expected to have 3 or 5 dimensions when key is empty');
    }
    if (query.dims.length === 5 && (query.dims[2] !== attributes.numHeads || query.dims[3] !== 3)) {
      throw new Error('Expect \'query\' shape (batch_size, kv_sequence_length, num_heads, 3, head_size) for packed kv');
    }

    qkvFormat = AttentionQkvFormat.QKV_BSN3H;
  }

  if (bias) {
    if (bias.dims.length !== 1) {
      throw new Error('Input \'bias\' is expected to have 1 dimension');
    }

    if (value) {
      if (query.dims.length === 5 && query.dims[3] === 2) {
        throw new Error('bias is not allowed for packed kv.');
      }
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
    throw new Error('Mask not supported');
  }

  let passPastInKv = false;
  let vHiddenSize = hiddenSize;
  if (value) {
    if (value.dims.length !== 3 && value.dims.length !== 4) {
      throw new Error('Input \'value\' is expected to have 3 or 4 dimensions')
    }

    if (query.dims[0] !== value.dims[0]) {
      throw new Error('Input \'query\' and \'value\' shall have same dim 0 (batch_size)')
    }

    if (value.dims.length === 3) {
      if (kvSequenceLength !== value.dims[1]) {
        throw new Error('Input \'key\' and \'value\' shall have the same dim 1 (kv_sequence_length)')
      }
      vHiddenSize = value.dims[2];
    } else {
      if (kvSequenceLength !== value.dims[2]) {
        throw new Error('Input \'past_key\' and \'past_value\' shall have the same dim 2 (kv_sequence_length)')
      }
      vHiddenSize = value.dims[1] * value.dims[3];
      passPastInKv = true;
    }
  }

  let totalSequenceLength = pastSequenceLength + kvSequenceLength;
  let broadcastResPosBias = false;
  // if (extraAddQk) {
  //   if (extraAddQk.dims[0] === 1) {
  //     broadcastResPosBias = true;
  //   }
  // }

  // if (bias) {
  //   throw new Error('bias is not supported');
  // }
  if (keyPaddingMask) {
    throw new Error('Key padding mask is not supported');
  }
  if (relativePositionBias) {
    throw new Error('extraAddQk is not supported');
  }
  if (pastKey) {
    throw new Error('pastKey is not supported');
  }
  if (pastValue) {
    throw new Error('pastValue is not supported');
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
    headSize,
    vHeadSize: Math.floor(vHiddenSize / attributes.numHeads),
    numHeads: attributes.numHeads,
    isUnidirectional: false,
    pastPresentShareBuffer: false,
    maskFilterValue: attributes.maskFilterValue,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias,
    passPastInKv,
    qkvFormat,
  };
};


export const parseMultiHeadAttentionAttributes = (attributes: AttentionAttrs): AttentionAttrs =>
    createAttributeWithCacheKey({...attributes});

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [0, 2, 1, 3]});
const packedWeightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [0, 2, 1, 3, 4]});

const addBiasTranspose =
    (context: ComputeContext, qkv: TensorView, bias: TensorView, batchSize: number, sequenceLength: number,
     hiddenSize: number, biasOffset: number) => {
      const addBiasTransposeMetadata = {
        name: 'addBiasTranspose',
        inputTypes: [GpuDataType.default, GpuDataType.default],
        cacheHint: JSON.stringify({batchSize, sequenceLength, hiddenSize, biasOffset}),
      };

      const outputShape = [batchSize, sequenceLength, hiddenSize];
      const outputSize = ShapeUtil.size(outputShape);

      const dataType = 'f32';
      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const biasOffset = ${biasOffset}u;
  const hiddenSize = ${hiddenSize}u;

  @group(0) @binding(0) var<storage, read> qkv: array<${dataType}>;
  @group(0) @binding(1) var<storage, read> bias: array<${dataType}>;
  @group(0) @binding(2) var<storage, read_write> qkv_with_bias: array<${dataType}>;

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let biasOffsetIdx = (global_idx % hiddenSize) + biasOffset;

    qkv_with_bias[global_idx] = qkv[global_idx] + bias[biasOffsetIdx];
  }`;

      return context.compute(
          {
            ...addBiasTransposeMetadata,
            outputs: [{dims: outputShape, dataType: DataType.float, gpuDataType: GpuDataType.default}],
            getShaderSource,
            dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
          },
          {inputs: [qkv, bias], outputs: [-1]})[0];
    };

const maybeTransposeToBNSHAndAddBias =
    (context: ComputeContext, batchSize: number, numHeads: number, sequenceLength: number, headSize: number,
     input: TensorView, bias?: TensorView, biasOffset?: number) => {
      // const newDims = [];

      let reshapedInput = input;
      if (!bias) {
        if (input.dims.length === 3) {
          reshapedInput = input.reshape([batchSize, sequenceLength, numHeads, headSize]);
        }
        return context.compute(
            {
              ...transposeProgramMetadata,
              cacheHint: weightTransposeAttribute.cacheKey,
              get: () => createTransposeProgramInfo(reshapedInput, weightTransposeAttribute.perm)
            },
            {inputs: [reshapedInput], outputs: [-1]})[0];
      } else {
        if (sequenceLength === 1) {
          throw new Error('AddBiasReshape is not implemented. Please export your model with packed QKV or KV');
        } else {
          reshapedInput =
              addBiasTranspose(context, input, bias, batchSize, sequenceLength, numHeads * headSize, biasOffset!);
          reshapedInput = reshapedInput.reshape([batchSize, sequenceLength, numHeads, headSize]);
          return context.compute(
              {
                ...transposeProgramMetadata,
                cacheHint: weightTransposeAttribute.cacheKey + biasOffset!.toString() + Math.random().toString(10),
                get: () => createTransposeProgramInfo(reshapedInput, weightTransposeAttribute.perm)
              },
              {inputs: [reshapedInput], outputs: [-1]})[0];
        }
      }
    };

// const getMaxComponents = (size: number) => {
//     if (size % 4 === 0) {
//         return 4;
//     } else if (size % 3 === 0) {
//         return 3;
//     } else if (size % 2 === 0) {
//         return 2;
//     }
//
//     return 1;
// };

const fillVector = (components?: number) => {
  if (!components || components === 1) {
    return 'f32(0)';
  }

  return `vec${components}<f32>(${new Array(components).fill(0).join(',')})`;
};

const computeAttentionProbsBSN3H =
    (context: ComputeContext, q: TensorView, key: TensorView, bias: TensorView|undefined,
     parameters: AttentionParameters, attributes: AttentionAttrs) => {
      const probsShape = [
        parameters.batchSize, parameters.sequenceLength, parameters.numHeads,
        parameters.kvSequenceLength + parameters.pastSequenceLength
      ];

      const components = undefined;  // getMaxComponents(parameters.headSize);
      const qInput = inputVariable('q', q.dataType, q.dims, components);
      const output = outputVariable('output', q.dataType, probsShape);

      const alpha = attributes.scale === 0 ? 1.0 / Math.sqrt(parameters.headSize) : attributes.scale;

      const unitsOfWork = ShapeUtil.size(probsShape);

      const M = parameters.sequenceLength;
      const N = parameters.totalSequenceLength;
      const K = parameters.headSize;

      // since we are multiplying Q with transposed K and headSize = vHeadSize,
      // we are multiplying Q head rows with K head rows for each head
      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K / (components || 1)}u;
  const numHeads: u32 = ${parameters.numHeads};
  const batchSize: u32 = ${parameters.batchSize};
  const alpha = f32(${alpha});
  const beta = 1.0;

  ${shaderHelper.declareVariables(qInput, output)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(unitsOfWork)}
    // batch and head index
    let batchIdx = global_idx / (M * N * numHeads);
    let headIdx = (global_idx / (M * N)) % numHeads;
    let qSequenceIdx = (global_idx / N) % ${parameters.sequenceLength};
    let kSequenceIdx = global_idx % (M * N) % ${parameters.totalSequenceLength};

    var headOffset = headIdx * ${parameters.headSize} * 3;

    var qOffset = qSequenceIdx * ${parameters.headSize} * numHeads * 3 + headOffset;
    var batchOffset = batchIdx * ${parameters.headSize} * numHeads * 3 * M;
    qOffset += batchOffset; // batch offset
    let kOffset = ${parameters.headSize}u + batchOffset + headOffset + kSequenceIdx * 
        ${parameters.headSize} * numHeads * 3;
    var value: ${qInput.type.storage} = ${fillVector(components)};
    for (var k: u32 = 0u; k<${K}u; k++) {
      value += q[k + qOffset] * q[k + kOffset];
    }

    let sum = ${sumVector('value', components)} * alpha;
    // value += beta * output[global_id.x]; // no mask
    output[global_idx] = sum;
  }`;

      const inputTypes = [1].map(_ => GpuDataType.default);

      const probs = context.compute(
          {
            name: 'computeAttentionProbsBSN3H',
            cacheHint: JSON.stringify(parameters),
            inputTypes,
            outputs: [{dims: probsShape, dataType: q.dataType, gpuDataType: GpuDataType.default}],
            getShaderSource,
            dispatchGroup: () => ({x: Math.ceil(unitsOfWork / 64 /* workgroup size */)})
          },
          {inputs: [q], outputs: [-1]})[0];

      computeInPlaceSoftmax(
          context, probs, parameters.batchSize * parameters.numHeads * parameters.sequenceLength,
          parameters.totalSequenceLength);

      return probs;
    };

const computeVxAttentionScoreBSN3H = (probs: TensorView, qkv: TensorView, params: AttentionParameters) => {
  const attentionScoreMatMulProgramData = {
    name: 'computeVxAttentionScore',
    inputTypes: [GpuDataType.default, GpuDataType.default],
    cacheHint: JSON.stringify(params),
  };

  const outputShape = [params.batchSize, params.sequenceLength, params.numHeads, params.vHeadSize];
  const outputSize = ShapeUtil.size(outputShape);

  const probsHelper = inputVariable('probs', probs.dataType, probs.dims);
  const qkvHelper = inputVariable('qkv', qkv.dataType, qkv.dims);
  const output = outputVariable('output', probs.dataType, outputShape);

  const dataType = 'f32';
  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${params.sequenceLength}u;
  const N: u32 = ${params.vHeadSize}u;
  const K: u32 = ${params.totalSequenceLength}u;
  const numHeads: u32 = ${params.numHeads}u;
  const batchSize: u32 = ${params.batchSize};

  ${shaderHelper.declareVariables(probsHelper, qkvHelper, output)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let batchIdx = global_idx / (M * N * numHeads);
    let headIdx = (global_idx / (M * N)) % numHeads;
    let probsSequenceIdx = (global_idx / N) % ${params.sequenceLength};

    let offsetA = probsSequenceIdx * ${params.headSize} * numHeads + batchIdx * ${params.headSize} * numHeads * M;
    
    var headOffset = headIdx * ${params.vHeadSize} * 3;
    var batchOffset = batchIdx * ${params.vHeadSize} * numHeads * 3 * M;
    

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<K; k++) {
      var vOffset = ${params.headSize * 2}u + batchOffset + headOffset + k * 
          ${params.headSize} * numHeads * 3;
      value += probs[offsetA + k] * qkv[vOffset];
    }
    output[global_idx] = f32(offsetA);
  }`;
  return {
    ...attentionScoreMatMulProgramData,
    outputs: [{dims: outputShape, dataType: DataType.float, gpuDataType: GpuDataType.default}],
    getShaderSource,
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
  };
};

export const applyPackedAttention =
    (context: ComputeContext, q: TensorView, k: TensorView, v: TensorView, maskIndex: TensorView|undefined,
     past: TensorView|undefined, pastKey: TensorView|undefined, pastValue: TensorView|undefined,
     relativePositionBias: TensorView|undefined, parameters: AttentionParameters, attributes: AttentionAttrs) => {
      const probs = computeAttentionProbsBSN3H(context, q, k, relativePositionBias, parameters, attributes);

      const attentionScoreMatMulProgramData = {
        name: 'PackedAttentionScore',
        inputTypes: [GpuDataType.default, GpuDataType.default],
        cacheHint: JSON.stringify(parameters) + JSON.stringify(attributes),
      };

      const attentionResult = context.compute(
          {
            ...attentionScoreMatMulProgramData,
            cacheHint: JSON.stringify(parameters),
            get: () => computeVxAttentionScoreBSN3H(probs, q, parameters)
          },
          {inputs: [probs, v || q], outputs: [-1]})[0];

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

export const multiHeadAttention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateInputs(context.inputs, attributes);

  if (context.inputs[0].dims.length === 5) {
    // transpose QKV from BSN3H to BNS3H
    return applyPackedAttention(
        context, context.inputs[0], context.inputs[1], context.inputs[2], context.inputs[4], undefined,
        context.inputs[6], context.inputs[7], context.inputs[5], params, attributes);
  }

  if (context.inputs[1]?.dims.length === 5) {
    // transpose Q from BSD (BSNH) to BNSH
    const Q = maybeTransposeToBNSHAndAddBias(
        context, params.batchSize, params.numHeads, params.sequenceLength, params.headSize, context.inputs[0],
        context.inputs[3], 0);

    // transpose KV from BLN2H to BNS2H
    const K = context.compute(
        {
          ...transposeProgramMetadata,
          cacheHint: weightTransposeAttribute.cacheKey,
          get: () => createTransposeProgramInfo(context.inputs[1], packedWeightTransposeAttribute.perm)
        },
        {inputs: [context.inputs[0]], outputs: [-1]})[0];
    return applyAttention(
        context, Q, K, context.inputs[2], context.inputs[4], undefined, context.inputs[6], context.inputs[7],
        context.inputs[5], params, attributes);
  }

  // applyAttention expects BNSH inputs
  const kvBNSH = context.inputs[1] && context.inputs[2] && context.inputs[1].dims.length === 4 &&
      context.inputs[2].dims.length === 4;

  const Q = maybeTransposeToBNSHAndAddBias(
      context, params.batchSize, params.numHeads, params.sequenceLength, params.headSize, context.inputs[0],
      context.inputs[3], 0);

  if (kvBNSH) {
    return applyAttention(
        context, Q, context.inputs[1], context.inputs[2], context.inputs[4], undefined, undefined, undefined,
        context.inputs[5], params, attributes);
  }

  const K = maybeTransposeToBNSHAndAddBias(
      context, params.batchSize, params.numHeads, params.kvSequenceLength, params.headSize, context.inputs[1],
      context.inputs[3], params.hiddenSize);

  const V = maybeTransposeToBNSHAndAddBias(
      context, params.batchSize, params.numHeads, params.kvSequenceLength, params.vHeadSize, context.inputs[2],
      context.inputs[3], 2 * params.hiddenSize);

  applyAttention(
      context, Q, K, V, context.inputs[4], undefined, context.inputs[6], context.inputs[7], context.inputs[5], params,
      attributes);
};
