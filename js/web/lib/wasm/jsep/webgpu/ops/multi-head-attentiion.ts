import {TensorView} from '../../tensor';
import {createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext} from '../types';

import {applyAttention, AttentionAttrs, AttentionMaskType, AttentionParameters, AttentionQkvFormat} from './attentiion';
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

  let qkvFormat = AttentionQkvFormat.Q_K_V_BSNH;
  if (key) {
    if (query.dims.length !== 3) {
      throw new Error('Input \'query\' is expected to have 3 dimensions when key is given');
    }
    if (key.dims.length < 3 || key.dims.length > 5) {
      throw new Error('Input \'key\' is expected to have 3, 4, or 5 dimensions');
    }
    if (query.dims[0] != key.dims[0]) {
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
      if (query.dims.length === 5 && query.dims[3] == 2) {
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

  if (bias) {
    throw new Error('bias is not supported');
  }
  if (keyPaddingMask) {
    throw new Error('Key padding mask is not supported');
  }
  if (relativePositionBias) {
    throw new Error('extraAddQk is not supported')
  }
  if (pastKey) {
    throw new Error('pastKey is not supported')
  }
  if (pastValue) {
    throw new Error('pastValue is not supported')
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
          throw new Error('AddBiasTranspose is not implemented. Please export your model with packed QKV or KV');
        }
      }
    };

const unpackQKV = (input1: TensorView, input2?: TensorView) => {
  const inputToSplit = input2 || input1;
  const dims = inputToSplit.dims;
  const offsetBase = dims[0] * dims[1] * dims[2] * dims[4] * 4;
  const Q = input1 || input2;

  if (input2) {
    const K = input2;
    const V = {...input2};
    V.offset = offsetBase;
    return [Q, K, V];
  }
  const K = {...inputToSplit};
  K.offset = offsetBase;
  const V = {...inputToSplit};
  V.offset = offsetBase * 2;

  return [Q, K, V];
};

export const multiHeadAttention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateInputs(context.inputs, attributes);
  if (context.inputs[0].dims.length === 5 || context.inputs[1]?.dims.length === 5) {
    const [Q, K, V] = unpackQKV(context.inputs[0], context.inputs[1]);
    return applyAttention(
        context, Q, K, V, context.inputs[4], undefined, context.inputs[6], context.inputs[7], context.inputs[5], params,
        attributes);
  }

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
