import { ComputeContext } from '../types';
import { TensorView } from '../../tensor';
import { createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata } from './transpose';
import {
  applyAttention,
  AttentionAttrs,
  AttentionMaskType,
  AttentionParameters,
  AttentionQkvFormat
} from './attentiion';

const validateInputs = (inputs: readonly TensorView[], attributes: AttentionAttrs): AttentionParameters => {
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
    : attributes.numHeads * query.dims[4];
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
    headSize: hiddenSize / attributes.numHeads,
    vHeadSize: vHiddenSize / attributes.numHeads,
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
  createAttributeWithCacheKey({ ...attributes });

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [0, 2, 1, 3]});
const maybeTransposeToBNSHAndAddBias = (context: ComputeContext, batchSize: number, numHeads: number,
  sequenceLength: number, headSize: number, input: TensorView, bias?: TensorView, biasOffset?: number) => {
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
      throw new Error('implement AddBiasReshape');
    } else {
      throw new Error('implement AddBiasTranspose');
    }
  }
};

export const multiHeadAttention = (context: ComputeContext, attributes: AttentionAttrs): void => {
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
