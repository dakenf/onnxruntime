import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from "../types";
import {TensorView} from "../../tensor";
import {DataType, tensorTypeToWsglType} from "../../../wasm-common";
import {ShapeUtil} from "../../util";
import {ShaderHelper} from "./common";
import {AttributeWithCacheKey, createAttributeWithCacheKey} from "../attribute-with-cache-key";

export interface InstanceNormAttributes extends AttributeWithCacheKey {
    epsilon: number;
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
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: InstanceNormAttributes):
        ProgramInfo => {
        const xShape = inputs[0].dims;
        const scale = inputs[1];
        const bias = inputs[2];

        const outputShape = xShape;
        const outputSize = ShapeUtil.size(outputShape);
        const axis = 2;
        const normCount = ShapeUtil.sizeToDimension(xShape, axis);
        const normSize = ShapeUtil.sizeFromDimension(xShape, axis);
        const C = xShape[1];

        const scaleSize = ShapeUtil.size(scale.dims);
        const biasSize = bias ? ShapeUtil.size(bias.dims) : 0;
        if (scaleSize !== normSize || (bias && biasSize !== normSize)) {
            throw new Error(`Size of X.shape()[axis:] == ${normSize}.
             Size of scale and bias (if provided) must match this. 
             Got scale size of ${scaleSize} and bias size of ${biasSize}`);
        }

        console.log('instance norm!', inputs, normCount, normSize, axis, attributes);
        const dataType = tensorTypeToWsglType(inputs[0].dataType);

        const workgroupSize = normCount > 128 ? 256 : 64;
        const getShaderSource = (shaderHelper: ShaderHelper) => `
  const C: u32 = ${C};
  const normSize: u32 = ${normSize};
  const normSizeTyped: ${dataType} = ${normSize};
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart(workgroupSize)}
    let offset = global_idx * normSize;
    if (offset + normSize >= ${outputSize}) { return; }
    var mean: ${dataType} = 0;
    var meanSquare: ${dataType} = 0;

    for (var h: u32 = 0u; h < normSize; h++) {
        mean = mean + x[h + offset];
        meanSquare = meanSquare + x[h + offset] * x[h + offset];
    }
    mean = mean / normSizeTyped;
    meanSquare = sqrt(meanSquare / normSizeTyped - mean * mean + epsilon);
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

    //meanDataOutput[global_idx] = mean;
    //invStdOutput[global_idx] = 1 / meanSquare;
  }`;
        return {
            ...metadata,
            outputs: [
                {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
                // {dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
                // {dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
            ],
            getShaderSource,
            dispatchGroup: () => ({x: Math.ceil(normCount / workgroupSize /* workgroup size */)})
        };
    };

export const parseInstanceNormAttributes = (attributes: Record<string, unknown>): InstanceNormAttributes =>
    createAttributeWithCacheKey(attributes as Omit<InstanceNormAttributes, keyof AttributeWithCacheKey>);

export const instanceNorm = (context: ComputeContext, attributes: InstanceNormAttributes): void => {
    validateInputs(context.inputs);

    const metadata = {
        name: 'InstanceNormalization',
        inputTypes: [GpuDataType.default, GpuDataType.default, GpuDataType.default],
        cacheHint: attributes.cacheKey,
    };

    context.compute(createInstanceNormProgramInfo(metadata, context.inputs, attributes));
};