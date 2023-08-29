// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env, InferenceSession, Tensor} from 'onnxruntime-common';

import {FSNode} from './binding/ort-wasm';
import {SerializableModeldata, SerializableSessionMetadata, SerializableTensor} from './proxy-messages';
import {setRunOptions} from './run-options';
import {setSessionOptions} from './session-options';
import {logLevelStringToEnum, tensorDataTypeEnumToString, tensorDataTypeStringToEnum, tensorTypeToTypedArrayConstructor} from './wasm-common';
import {getInstance} from './wasm-factory';
import {allocWasmString, checkLastError} from './wasm-utils';

/**
 * get the input/output count of the session.
 * @param sessionHandle the handle representing the session. should be non-zero.
 * @returns a tuple including 2 numbers, representing the input count and output count.
 */
const getSessionInputOutputCount = (sessionHandle: number): [number, number] => {
  const wasm = getInstance();
  const stack = wasm.stackSave();
  try {
    const ptrSize = wasm.PTR_SIZE;
    const dataOffset = wasm.stackAlloc(2 * ptrSize);
    const errorCode = wasm._OrtGetInputOutputCount(sessionHandle, dataOffset, dataOffset + ptrSize);
    if (errorCode !== 0) {
      checkLastError('Can\'t get session input/output count.');
    }
    return [wasm.getValue(dataOffset, '*'), wasm.getValue(dataOffset + ptrSize, '*')];
  } finally {
    wasm.stackRestore(stack);
  }
};

/**
 * initialize ORT environment.
 * @param numThreads SetGlobalIntraOpNumThreads(numThreads)
 * @param loggingLevel CreateEnv(static_cast<OrtLoggingLevel>(logging_level))
 */
const initOrt = (numThreads: number, loggingLevel: number): void => {
  const errorCode = getInstance()._OrtInit(numThreads, loggingLevel);
  if (errorCode !== 0) {
    checkLastError('Can\'t initialize onnxruntime.');
  }
};

/**
 * intialize runtime environment.
 * @param env passed in the environment config object.
 */
export const initRuntime = async(env: Env): Promise<void> => {
  // init ORT
  initOrt(32, logLevelStringToEnum(env.logLevel));

  if (!BUILD_DEFS.DISABLE_WEBGPU) {
    // init JSEP if available

    // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-var-requires
    const initJsep = require('./jsep/init').init;
    await initJsep(getInstance(), env);
  }
};

/**
 *  tuple elements are: InferenceSession ID; inputNamesUTF8Encoded; outputNamesUTF8Encoded
 */
type SessionMetadata = [number, number[], number[]];

const activeSessions = new Map<number, SessionMetadata>();

/**
 * allocate the memory and memcpy the model bytes, preparing for creating an instance of InferenceSession.
 * @returns a 3-elements tuple - the pointer, size of the allocated buffer, and optional weights.pb FS node
 */
export const createSessionAllocate = (model: Uint8Array, weights?: ArrayBuffer): [number, number, FSNode?] => {
  const wasm = getInstance();
  const modelDataOffset = wasm._malloc(model.byteLength);
  wasm.HEAPU8.set(model, modelDataOffset);

  let weightsFile: FSNode|undefined;
  if (weights) {
    weightsFile = wasm.FS.create('/home/web_user/weights.pb');
    weightsFile.contents = weights;
    weightsFile.usedBytes = weights.byteLength;
    wasm.FS.chdir('/home/web_user');
  }
  return [modelDataOffset, model.byteLength, weightsFile];
};

/**
 * create an inference session using the prepared buffer containing the model data.
 * @param modelData a 2-elements tuple containing the pointer and size of the model data buffer.
 * @param options an optional session options object.
 * @returns a 3-elements tuple containing [session handle, input names, output names]
 */
export const createSessionFinalize =
    (modelData: SerializableModeldata, options?: InferenceSession.SessionOptions): SerializableSessionMetadata => {
      const wasm = getInstance();

      let sessionHandle = 0;
      let sessionOptionsHandle = 0;
      let allocs: number[] = [];
      const inputNamesUTF8Encoded = [];
      const outputNamesUTF8Encoded = [];

      try {
        [sessionOptionsHandle, allocs] = setSessionOptions(options);

        sessionHandle = wasm._OrtCreateSession(modelData[0], modelData[1], sessionOptionsHandle);
        if (sessionHandle === 0) {
          checkLastError('Can\'t create a session.');
        }

        const [inputCount, outputCount] = getSessionInputOutputCount(sessionHandle);

        const inputNames = [];
        const outputNames = [];
        for (let i = 0; i < inputCount; i++) {
          const name = wasm._OrtGetInputName(sessionHandle, i);
          if (name === 0) {
            checkLastError('Can\'t get an input name.');
          }
          inputNamesUTF8Encoded.push(name);
          inputNames.push(wasm.UTF8ToString(name));
        }
        for (let i = 0; i < outputCount; i++) {
          const name = wasm._OrtGetOutputName(sessionHandle, i);
          if (name === 0) {
            checkLastError('Can\'t get an output name.');
          }
          outputNamesUTF8Encoded.push(name);
          outputNames.push(wasm.UTF8ToString(name));
        }

        activeSessions.set(sessionHandle, [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded]);
        return [sessionHandle, inputNames, outputNames];
      } catch (e) {
        inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
        outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));

        if (sessionHandle !== 0) {
          wasm._OrtReleaseSession(sessionHandle);
        }
        throw e;
      } finally {
        wasm._free(modelData[0]);
        if (sessionOptionsHandle !== 0) {
          wasm._OrtReleaseSessionOptions(sessionOptionsHandle);
        }
        allocs.forEach(alloc => wasm._free(alloc));
        if (modelData[2]) {
          wasm.FS.unlink('/home/web_user/weights.pb');
        }
      }
    };


/**
 * create an instance of InferenceSession.
 * @returns the metadata of InferenceSession. 0-value handle for failure.
 */
export const createSession =
    async(model: Uint8Array, options?: InferenceSession.SessionOptions): Promise<SerializableSessionMetadata> => {
  const modelData: SerializableModeldata = createSessionAllocate(model);
  return createSessionFinalize(modelData, options);
};

export const releaseSession = (sessionId: number): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error(`cannot release session. invalid session id: ${sessionId}`);
  }
  const [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded] = session;

  inputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
  outputNamesUTF8Encoded.forEach(buf => wasm._OrtFree(buf));
  wasm._OrtReleaseSession(sessionHandle);
  activeSessions.delete(sessionId);
};

/**
 * perform inference run
 */
export const run = async(
    sessionId: number, inputIndices: number[], inputs: SerializableTensor[], outputIndices: number[],
    options: InferenceSession.RunOptions): Promise<SerializableTensor[]> => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error(`cannot run inference. invalid session id: ${sessionId}`);
  }
  const [sessionHandle, inputNamesUTF8Encoded, outputNamesUTF8Encoded] = session;

  const inputCount = inputIndices.length;
  const outputCount = outputIndices.length;

  let runOptionsHandle = 0;
  let runOptionsAllocs: number[] = [];

  const inputValues: number[] = [];
  const inputAllocs: number[] = [];

  try {
    [runOptionsHandle, runOptionsAllocs] = setRunOptions(options);
    const ptrSize = wasm.PTR_SIZE;
    // create input tensors
    for (let i = 0; i < inputCount; i++) {
      const dataType = inputs[i][0];
      const dims = inputs[i][1];
      const data = inputs[i][2];

      let dataOffset: number;
      let dataByteLength: number;

      if (Array.isArray(data)) {
        // string tensor
        dataByteLength = 4 * data.length;
        dataOffset = wasm._malloc(dataByteLength);
        inputAllocs.push(dataOffset);
        let dataIndex = dataOffset / 4;
        for (let i = 0; i < data.length; i++) {
          if (typeof data[i] !== 'string') {
            throw new TypeError(`tensor data at index ${i} is not a string`);
          }
          wasm.HEAPU32[dataIndex++] = allocWasmString(data[i], inputAllocs);
        }
      } else {
        dataByteLength = data.byteLength;
        dataOffset = wasm._malloc(dataByteLength);
        inputAllocs.push(dataOffset);
        wasm.HEAPU8.set(new Uint8Array(data.buffer, data.byteOffset, dataByteLength), dataOffset);
      }

      const stack = wasm.stackSave();
      const dimsOffset = wasm.stackAlloc(ptrSize * dims.length);
      try {
        dims.forEach((d, index) => wasm.setValue(dimsOffset + (index * ptrSize), d, '*'));
        const tensor = wasm._OrtCreateTensor(
            tensorDataTypeStringToEnum(dataType), dataOffset, dataByteLength, dimsOffset, dims.length);
        if (tensor === 0) {
          checkLastError(`Can't create tensor for input[${i}].`);
        }
        inputValues.push(tensor);
      } finally {
        wasm.stackRestore(stack);
      }
    }

    const beforeRunStack = wasm.stackSave();
    const inputValuesOffset = wasm.stackAlloc(inputCount * ptrSize);
    const inputNamesOffset = wasm.stackAlloc(inputCount * ptrSize);
    const outputValuesOffset = wasm.stackAlloc(outputCount * ptrSize);
    const outputNamesOffset = wasm.stackAlloc(outputCount * ptrSize);

    try {
      let inputValuesIndex = inputValuesOffset / 8;
      let inputNamesIndex = inputNamesOffset / 8;
      let outputValuesIndex = outputValuesOffset / 8;
      let outputNamesIndex = outputNamesOffset / 8;
      for (let i = 0; i < inputCount; i++) {
        wasm.HEAPU64[inputValuesIndex++] = BigInt(inputValues[i]);
        wasm.HEAPU64[inputNamesIndex++] = BigInt(inputNamesUTF8Encoded[inputIndices[i]]);
      }
      for (let i = 0; i < outputCount; i++) {
        wasm.HEAPU64[outputValuesIndex++] = BigInt(0);
        wasm.HEAPU64[outputNamesIndex++] = BigInt(outputNamesUTF8Encoded[outputIndices[i]]);
      }

      // support RunOptions
      let errorCode = wasm._OrtRun(
          sessionHandle, inputNamesOffset, inputValuesOffset, inputCount, outputNamesOffset, outputCount,
          outputValuesOffset, runOptionsHandle);

      // eslint-disable-next-line @typescript-eslint/naming-convention
      const runPromise = wasm.jsepRunPromise;
      if (runPromise && typeof runPromise.then !== 'undefined') {
        errorCode = await runPromise;
      }

      const output: SerializableTensor[] = [];

      if (errorCode !== 0) {
        checkLastError('failed to call OrtRun().');
      }
      const ptrSize = 8;
      for (let i = 0; i < outputCount; i++) {
        const tensor = wasm.getValue(outputValuesOffset + i * ptrSize, '*');

        const beforeGetTensorDataStack = wasm.stackSave();
        // stack allocate 4 pointer value
        const tensorDataOffset = wasm.stackAlloc(4 * 8);

        let type: Tensor.Type|undefined, dataOffset = 0;
        try {
          errorCode = wasm._OrtGetTensorData(
              tensor, tensorDataOffset, tensorDataOffset + ptrSize, tensorDataOffset + ptrSize * 2,
              tensorDataOffset + ptrSize * 3);
          if (errorCode !== 0) {
            checkLastError(`Can't access output tensor data on index ${i}.`);
          }

          const dataType = wasm.getValue(tensorDataOffset, '*');
          dataOffset = wasm.getValue(tensorDataOffset + ptrSize, '*');
          const dimsOffset = wasm.getValue(tensorDataOffset + ptrSize * 2, '*');
          const dimsLength = wasm.getValue(tensorDataOffset + ptrSize * 3, '*');
          const dims = [];
          for (let i = 0; i < dimsLength; i++) {
            dims.push(wasm.getValue(dimsOffset + i * ptrSize, '*'));
          }
          wasm._OrtFree(dimsOffset);

          const size = dims.length === 0 ? 1 : dims.reduce((a, b) => a * b);
          type = tensorDataTypeEnumToString(dataType);
          if (type === 'string') {
            const stringData: string[] = [];
            let dataIndex = dataOffset / 4;
            for (let i = 0; i < size; i++) {
              const offset = wasm.HEAPU32[dataIndex++];
              const maxBytesToRead = i === size - 1 ? undefined : wasm.HEAPU32[dataIndex] - offset;
              stringData.push(wasm.UTF8ToString(offset, maxBytesToRead));
            }
            output.push([type, dims, stringData]);
          } else {
            const typedArrayConstructor = tensorTypeToTypedArrayConstructor(type);
            const data = new typedArrayConstructor(size);
            new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                .set(wasm.HEAPU8.subarray(dataOffset, dataOffset + data.byteLength));
            output.push([type, dims, data]);
          }
        } finally {
          wasm.stackRestore(beforeGetTensorDataStack);
          if (type === 'string' && dataOffset) {
            wasm._free(dataOffset);
          }
          wasm._OrtReleaseTensor(tensor);
        }
      }

      return output;
    } finally {
      wasm.stackRestore(beforeRunStack);
    }
  } finally {
    inputValues.forEach(v => wasm._OrtReleaseTensor(v));
    inputAllocs.forEach(p => wasm._free(p));

    if (runOptionsHandle !== 0) {
      wasm._OrtReleaseRunOptions(runOptionsHandle);
    }
    runOptionsAllocs.forEach(p => wasm._free(p));
  }
};

/**
 * end profiling
 */
export const endProfiling = (sessionId: number): void => {
  const wasm = getInstance();
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw new Error('invalid session id');
  }
  const sessionHandle = session[0];

  // profile file name is not used yet, but it must be freed.
  const profileFileName = wasm._OrtEndProfiling(sessionHandle);
  if (profileFileName === 0) {
    checkLastError('Can\'t get an profile file name.');
  }
  wasm._OrtFree(profileFileName);
};

export const extractTransferableBuffers = (tensors: readonly SerializableTensor[]): ArrayBufferLike[] => {
  const buffers: ArrayBufferLike[] = [];
  for (const tensor of tensors) {
    const data = tensor[2];
    if (!Array.isArray(data) && data.buffer) {
      buffers.push(data.buffer);
    }
  }
  return buffers;
};
