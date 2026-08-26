let transformersMod = null;
const pipelines = {};
let devicePromise = null;
let jsDelivrCache = null;

const HF_REMOTE_HOST = 'https://huggingface.co/';
const HF_REMOTE_PATH_TEMPLATE = '{model}/resolve/{revision}/';

async function pickDevice() {
  if (!('gpu' in self.navigator)) return 'wasm';
  try {
    const adapter = await self.navigator.gpu.requestAdapter();
    return adapter ? 'webgpu' : 'wasm';
  } catch {
    return 'wasm';
  }
}

// Model weight URLs always look like ".../onnx/<name>.onnx" or
// ".../onnx/<name>.onnx_data[_N]" , the only files that are ever split
// into chunks in the CDN mirror. Everything else (tokenizer.json,
// config.json, etc.) is a normal single small file.
function looksLikeModelWeightFile(url) {
  return /\/onnx\/[^/]+\.onnx(_data(_\d+)?)?$/.test(url);
}

async function sha256Hex(buf) {
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// A Cache-API-shaped object (match/put) that transformers.js will use
// in place of its own browser cache when loading from the CDN mirror.
// For ordinary small files it's a thin pass-through to a real Cache
// Storage bucket. For a chunked weight file (recognized by a sibling
// "<file>.manifest.json"), it lazily fetches each chunk, verifies its
// SHA-256 against the manifest, and streams the bytes back so
// transformers.js's own download-progress UI keeps working , then
// stashes the fully-assembled file in the real cache in the background
// so the next load of the same model is instant.
function getJsDelivrCache() {
  if (jsDelivrCache) return jsDelivrCache;
  const realCachePromise = ('caches' in self)
    ? self.caches.open('palaver-jsdelivr-v1').catch(() => null)
    : Promise.resolve(null);
  jsDelivrCache = {
    async match(key) {
      const realCache = await realCachePromise;
      if (realCache) {
        try {
          const hit = await realCache.match(key);
          if (hit) return hit;
        } catch { /* ignore cache read errors */ }
      }
      if (!looksLikeModelWeightFile(key)) return undefined;
      let manifest;
      try {
        const mRes = await fetch(key + '.manifest.json');
        if (!mRes.ok) return undefined; // not a chunked file , fall through to a normal fetch
        manifest = await mRes.json();
      } catch {
        return undefined;
      }
      const baseUrl = key.slice(0, key.lastIndexOf('/') + 1);
      // For files under ~200MB, accumulate chunks so we can stash the
      // assembled blob in Cache API for instant offline reloads. For
      // larger files (Gemma 4 E2B is ~3.4GB), accumulating every chunk
      // into a second buffer while the ONNX runtime is also loading
      // them causes peak memory to blow the tab , so the stream passes
      // through directly without buffering, and we skip the cache write.
      const LARGE_FILE_THRESHOLD = 200 * 1024 * 1024;
      const shouldCollect = manifest.totalSize <= LARGE_FILE_THRESHOLD;
      const collected = shouldCollect ? [] : null;
      let idx = 0;
      const stream = new ReadableStream({
        async pull(controller) {
          if (idx >= manifest.parts.length) {
            controller.close();
            if (shouldCollect) {
              realCachePromise.then(async (cacheToStore) => {
                if (!cacheToStore) return;
                try {
                  const fullBlob = new Blob(collected);
                  await cacheToStore.put(key, new Response(fullBlob, {
                    headers: { 'content-length': String(manifest.totalSize), 'content-type': 'application/octet-stream' },
                  }));
                } catch { /* best-effort caching only */ }
              });
            }
            return;
          }
          const part = manifest.parts[idx++];
          try {
            const res = await fetch(baseUrl + part.name);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const buf = new Uint8Array(await res.arrayBuffer());
            const hash = await sha256Hex(buf.buffer);
            if (hash !== part.sha256) throw new Error('checksum mismatch for ' + part.name);
            if (shouldCollect) collected.push(buf);
            controller.enqueue(buf);
          } catch (err) {
            controller.error(new Error('CDN chunk fetch failed (' + part.name + '): ' + err.message));
          }
        },
      });
      return new Response(stream, {
        headers: { 'content-length': String(manifest.totalSize), 'content-type': 'application/octet-stream' },
      });
    },
    async put(key, response) {
      const realCache = await realCachePromise;
      if (!realCache) return;
      try { await realCache.put(key, response); } catch { /* best-effort caching only */ }
    },
  };
  return jsDelivrCache;
}

// ---- JSPI ORT session loading (Blob-backed external data) ----

let jspiOrt = null;

// Create a disk-backed Blob from a chunked file.  Each 18 MB part is
// downloaded, verified, and enqueued into a ReadableStream.  The stream
// is consumed by Response.blob(), which on modern Chrome stores the
// result on disk , keeping peak JS heap at ~18 MB (one chunk at a time)
// instead of the full file size.  ORT's JSPI async loader then reads
// byte ranges from this Blob on demand during session creation.
async function makeChunkedBlob(manifest, baseUrl, onProgress, fileName) {
  const totalSize = manifest.totalSize;
  let loaded = 0;
  const stream = new ReadableStream({
    async pull(controller) {
      for (const part of manifest.parts) {
        const res = await fetch(baseUrl + part.name);
        if (!res.ok) throw new Error('CDN chunk fetch failed (' + part.name + '): HTTP ' + res.status);
        const buf = new Uint8Array(await res.arrayBuffer());
        loaded += buf.byteLength;
        if (onProgress) onProgress({ status: 'progress', file: fileName, loaded, total: totalSize });
        controller.enqueue(buf);
      }
      controller.close();
    },
  });
  return await new Response(stream, {
    headers: { 'content-length': String(totalSize), 'content-type': 'application/octet-stream' },
  }).blob();
}

// JSPI ORT sessions return tensors from a separate WASM instance that
// fails transformers.js's isONNXTensor() check (instanceof against the
// bundled asyncify ORT's Tensor class).  This wrapper intercepts .run()
// and re-creates each output tensor using the standard ORT's Tensor
// constructor so that replaceTensors() and validateInputs() accept them.
function wrapJspiSession(session) {
  const origRun = session.run.bind(session);
  session.run = async function(feeds) {
    const output = await origRun(feeds);
    // Access the standard ORT Tensor class via the global symbol that
    // transformers.js registered at import time.
    const stdOrt = globalThis[Symbol.for('onnxruntime')];
    const StdTensor = stdOrt && stdOrt.Tensor;
    if (!StdTensor) return output;
    const wrapped = {};
    for (const [key, val] of Object.entries(output)) {
      if (val && typeof val === 'object' && typeof val.type === 'string' && val.dims) {
        wrapped[key] = new StdTensor(val.type, val.data, val.dims);
        if (val.location) wrapped[key].location = val.location;
      } else {
        wrapped[key] = val;
      }
    }
    return wrapped;
  };
  return session;
}

// Create an ORT InferenceSession using the JSPI build, with the ONNX graph
// as a small ArrayBuffer and external data backed by a disk-backed Blob.
async function createSessionWithBlobData(graphUrl, dataUrl, dataFileName, sessionName, execProviders, onProgress) {
  onProgress({ status: 'progress', file: sessionName + '.onnx', loaded: 0, total: 0 });

  const graphRes = await fetch(graphUrl);
  if (!graphRes.ok) throw new Error('Failed to fetch ' + graphUrl + ': HTTP ' + graphRes.status);
  const graphBuf = new Uint8Array(await graphRes.arrayBuffer());
  onProgress({ status: 'progress', file: sessionName + '.onnx', loaded: graphBuf.byteLength, total: graphBuf.byteLength });

  let externalData = [];
  if (dataUrl) {
    const manifestRes = await fetch(dataUrl + '.manifest.json');
    if (manifestRes.ok) {
      const manifest = await manifestRes.json();
      const baseUrl = dataUrl.slice(0, dataUrl.lastIndexOf('/') + 1);
      const blob = await makeChunkedBlob(manifest, baseUrl, onProgress, dataFileName);
      externalData = [{ path: dataFileName, data: blob }];
    }
  }

  let session = await jspiOrt.InferenceSession.create(graphBuf, {
    executionProviders: execProviders,
    logSeverityLevel: 3,
    externalData,
  });
  session.config = { dtype: 'q4f16', device: execProviders[0] === 'webgpu' ? 'webgpu' : 'wasm' };
  return wrapJspiSession(session);
}

async function loadPipeline(modelId, dtype, device, cdnFolder, task, onProgress) {
  if (!transformersMod) {
    transformersMod = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
    transformersMod.env.backends.onnx.wasm.numThreads = 1;
    transformersMod.env.backends.onnx.wasm.wasmMemory = {
      initial: 6144,
      maximum: 65536,
      growth: true,
    };
  }
  let pipelineModelId = modelId;
  if (cdnFolder) {
    transformersMod.env.remoteHost = 'https://cdn.jsdelivr.net/gh/Sm0keSkreen/palaver-models@main/';
    transformersMod.env.remotePathTemplate = '{model}/';
    transformersMod.env.useCustomCache = true;
    transformersMod.env.customCache = getJsDelivrCache();
    pipelineModelId = cdnFolder;
  } else {
    transformersMod.env.remoteHost = HF_REMOTE_HOST;
    transformersMod.env.remotePathTemplate = HF_REMOTE_PATH_TEMPLATE;
    transformersMod.env.useCustomCache = false;
    transformersMod.env.customCache = null;
  }
  // Gemma 4 is a multimodal any-to-any model , transformers.js has no
  // pipeline('text-generation', ...) support for it, so it's loaded via
  // its own processor + model classes instead (mirrors the model
  // card's documented Transformers.js usage exactly).
  if (task === 'gemma4') {
    // Load processor and config via transformers.js (lightweight)
    const [processor, config] = await Promise.all([
      transformersMod.AutoProcessor.from_pretrained(pipelineModelId),
      transformersMod.AutoConfig.from_pretrained(pipelineModelId),
    ]);

    // JSPI ORT is a separate WASM instance used only for session
    // creation with Blob-backed external data.  We MUST NOT set it as
    // the global ORT because its WASM build is missing kernels (e.g.
    // GatherBlockQuantized) that the standard asyncify build has.
    // Instead, we wrap each JSPI session's .run() to convert output
    // tensors into objects that transformers.js's isONNXTensor() and
    // Tensor constructor recognize.
    if (!jspiOrt) {
      jspiOrt = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.jspi.min.mjs');
      jspiOrt.env.wasm.numThreads = 1;
      jspiOrt.env.wasm.wasmMemory = { initial: 6144, maximum: 65536, growth: true };
    }

    // Determine execution providers
    const execProviders = device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];

    // Session name → { graphFile, dataFile } mapping
    const suffix = '_q4f16';
    const sessionDefs = {
      embed_tokens:         { graphFile: 'embed_tokens' + suffix + '.onnx',         dataFile: 'embed_tokens' + suffix + '.onnx_data' },
      decoder_model_merged: { graphFile: 'decoder_model_merged' + suffix + '.onnx', dataFile: 'decoder_model_merged' + suffix + '.onnx_data' },
      vision_encoder:       { graphFile: 'vision_encoder' + suffix + '.onnx',       dataFile: 'vision_encoder' + suffix + '.onnx_data' },
      audio_encoder:        { graphFile: 'audio_encoder' + suffix + '.onnx',        dataFile: 'audio_encoder' + suffix + '.onnx_data' },
    };

    const sessions = {};
    for (const [name, def] of Object.entries(sessionDefs)) {
      const graphUrl = 'https://cdn.jsdelivr.net/gh/Sm0keSkreen/palaver-models@main/' + pipelineModelId + '/onnx/' + def.graphFile;
      const dataUrl  = 'https://cdn.jsdelivr.net/gh/Sm0keSkreen/palaver-models@main/' + pipelineModelId + '/onnx/' + def.dataFile;
      sessions[name] = await createSessionWithBlobData(graphUrl, dataUrl, def.dataFile, name, execProviders, onProgress);
    }

    // Build the model from config + our sessions (skips transformers.js
    // session loading which would materialize all external data as Uint8Array)
    const model = new transformersMod.Gemma4ForConditionalGeneration(config, sessions, { generation_config: {} });
    return { isGemma4: true, processor, model };
  }
  return transformersMod.pipeline('text-generation', pipelineModelId, {
    dtype, device, progress_callback: onProgress,
  });
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'loadModel') {
    const { key, modelId, dtype, cdnFolder, task } = msg;
    try {
      if (!devicePromise) devicePromise = pickDevice();
      let device = await devicePromise;
      const files = {};
      const onProgress = (p) => {
        if (p.status === 'progress' && p.file) {
          files[p.file] = p;
          let loaded = 0, total = 0;
          for (const f of Object.values(files)) { loaded += f.loaded || 0; total += f.total || 0; }
          self.postMessage({ type: 'loadProgress', key, pct: total ? Math.round((loaded / total) * 100) : 0 });
        }
      };
      let generator;
      try {
        generator = await loadPipeline(modelId, dtype, device, cdnFolder, task, onProgress);
      } catch (err) {
        if (device !== 'wasm') {
          device = 'wasm';
          generator = await loadPipeline(modelId, dtype, 'wasm', cdnFolder, task, onProgress);
        } else {
          throw err;
        }
      }
      pipelines[key] = generator;
      self.postMessage({ type: 'loadDone', key });
    } catch (err) {
      self.postMessage({ type: 'loadError', key, message: String((err && err.message) || err) });
    }
  } else if (msg.type === 'generate') {
    const { reqId, key, messages, genOpts } = msg;
    try {
      const pipe = pipelines[key];
      if (!pipe) throw new Error('model not loaded: ' + key);
      if (pipe.isGemma4) {
        // Gemma 4's own model card shows message content as an array of
        // typed parts ({ type: 'text', text: ... }), but that shape
        // actually throws inside this model's chat template ("Unknown
        // ArrayValue filter: trim") , live-tested against the real
        // processor before shipping this. Plain string content (the
        // same shape every other model in this app already uses) works
        // correctly, so messages are passed through unchanged.
        const { processor, model } = pipe;
        // Gemma 4's docs: thinking is triggered by putting a literal
        // "<|think|>" token at the start of the system prompt, plus
        // passing enable_thinking to the template , both are needed,
        // one alone isn't enough per the model card.
        const thinking = !!genOpts.thinking;
        const chatMessages = thinking
          ? messages.map((m, idx) => (idx === 0 && m.role === 'system')
            ? { role: m.role, content: '<|think|>\n' + m.content }
            : m)
          : messages;
        const prompt = processor.apply_chat_template(chatMessages, {
          add_generation_prompt: true,
          enable_thinking: thinking,
        });
        const inputs = await processor(prompt, null, null, { add_special_tokens: false });
        const streamer = new transformersMod.TextStreamer(processor.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (chunk) => {
            self.postMessage({ type: 'genChunk', reqId, text: chunk });
          },
        });
        // Google's documented sampling defaults for Gemma 4, not this
        // app's usual temperature/top_p , do_sample still comes from
        // genOpts so Stop behaves the same as other models. max_new_tokens
        // gets a bigger floor when thinking is on: the reasoning trace
        // itself consumes part of that budget, and without headroom a
        // long thought could eat the whole thing and leave no tokens
        // left for the actual answer.
        const outputs = await model.generate({
          ...inputs,
          max_new_tokens: thinking ? Math.max(genOpts.max_new_tokens, 1536) : genOpts.max_new_tokens,
          do_sample: genOpts.do_sample,
          temperature: 1.0,
          top_p: 0.95,
          top_k: 64,
          streamer,
        });
        const decoded = processor.batch_decode(
          outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
          { skip_special_tokens: true },
        );
        self.postMessage({ type: 'genDone', reqId, text: decoded[0] });
        return;
      }
      const streamer = new transformersMod.TextStreamer(pipe.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk) => {
          self.postMessage({ type: 'genChunk', reqId, text: chunk });
        },
      });
      const thinking = !!genOpts.thinking;
      const wireMessages = thinking
        ? messages.map((m, idx) => (idx === 0 && m.role === 'system')
          ? { role: m.role, content: '<|think|>\n' + m.content }
          : m)
        : messages;
      const result = await pipe(wireMessages, Object.assign({}, genOpts, { streamer }));
      let text = result && result[0] && result[0].generated_text;
      if (Array.isArray(text)) text = (text[text.length - 1] && text[text.length - 1].content) || '';
      self.postMessage({ type: 'genDone', reqId, text });
    } catch (err) {
      self.postMessage({ type: 'genError', reqId, message: String((err && err.message) || err) });
    }
  }
};
