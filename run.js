#!/usr/bin/env node
/**
 * run.js — CLI entrypoint for the multimodal agent pipeline.
 * No browser, no server, no human needed.
 *
 * Usage:
 *   node run.js --config run.config.json
 *   node run.js --desc "build a REST API" --stack "Node.js, Express" --out output/api.js
 *
 * API keys from env vars:
 *   OPENROUTER_KEY   (required)
 *
 * Polish is handled by Claude Code CLI (`claude` must be in PATH).
 */

'use strict';
const https  = require('https');
const fs     = require('fs');
const path   = require('path');
const crypto = require('crypto');
const MAX_LEARNING_LOOPS = 1; // single pass to avoid overwork
const MAX_TIMEOUT_RETRIES = 2; // allow two retries before giving up
const MAX_CONTINUATION_PASSES = 2; // allow extra continuation for truncated code
const REQUEST_TIMEOUT_MS = 90000; // free models can be slow — 90s per HTTP request
const DEFAULT_CONCISE_CHAR_LIMIT = 2200;
const MODEL_COOLDOWN_MS = 10 * 60 * 1000;
const BLOCKED_PROVIDERS = ['anthropic/', 'openai/']; // never use paid Claude/OpenAI
const ARCHITECT_TIMEOUT_MS = 90000;
const AGENT_TIMEOUT_MS = 150000; // 2.5 min per agent — large coding tasks need time
const REVIEWER_TIMEOUT_MS = 120000;

// Preferred free models ordered by reliability (tried and verified to work)
const PREFERRED_MODEL_PREFIXES = [
  'google/gemini',
  'qwen/qwen',
  'meta-llama/llama-3',
  'mistralai/mistral',
  'baidu/cobuddy',       // free, reliable architect+agent fallback
  'openrouter/owl',      // free, works end-to-end when others rate-limited
  'nousresearch',
  'microsoft/phi',
  'deepseek',
  // Note: inclusionai/ling-2.6-1t is paid (requires credits) — removed
];
const modelCooldownUntil = new Map();

function wantsConciseOutput(text) {
  const s = String(text || '').toLowerCase();
  return s.includes('concise') || s.includes('brief') || s.includes('short') || s.includes('summary');
}

function extractRequestedLineCount(text) {
  const s = String(text || '').toLowerCase();
  const m = s.match(/\b(\d+)\s*-\s*line\b|\b(\d+)\s*line\b/);
  const n = Number((m && (m[1] || m[2])) || 0);
  if (!Number.isFinite(n) || n <= 0) return null;
  return Math.min(n, 20);
}

function enforceConciseMarkdown(text, maxChars = DEFAULT_CONCISE_CHAR_LIMIT) {
  const raw = String(text || '').trim();
  if (!raw) return raw;
  if (raw.length <= maxChars) return raw;
  const lines = raw.split(/\r?\n/);
  const kept = [];
  let total = 0;
  for (const line of lines) {
    const next = total + line.length + 1;
    if (next > maxChars) break;
    kept.push(line);
    total = next;
  }
  // Clean ending noise and hard-stop with explicit truncation marker.
  const compact = kept.join('\n').replace(/\n{3,}/g, '\n\n').trim();
  return `${compact}\n\n[Truncated for conciseness]`;
}

function enforceExactLineCount(text, lineCount) {
  if (!lineCount || lineCount <= 0) return String(text || '').trim();
  const cleaned = String(text || '')
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(Boolean)
    .map(l => l.replace(/^[-*]\s+/, ''));
  const out = [];
  for (const line of cleaned) {
    if (out.length >= lineCount) break;
    out.push(line);
  }
  while (out.length < lineCount) out.push('N/A');
  return out.join('\n');
}

function buildLearningProtocolPrompt(taskText, extraInstructions = '') {
  const task = String(taskText || '').trim();
  const extra = String(extraInstructions || '').trim();
  return [
    'Execution rules:',
    '- Keep output concise and directly implementable.',
    '- Do not expand scope beyond the task.',
    '- If details are missing, make one practical assumption and proceed.',
    '',
    'Task:',
    task,
    ...(extra ? ['', 'Additional instructions:', extra] : []),
  ].join('\n');
}

function buildRefinementPrompt(taskPrompt, priorOutput, priorMeta = '') {
  return [
    'Final refinement pass:',
    '- Fix only missing or broken parts.',
    '- Keep output short and focused.',
    '',
    'Original task:',
    String(taskPrompt || '').trim(),
    '',
    'Prior output:',
    String(priorOutput || '').trim() || '(empty)',
    '',
    'Context from previous pass:',
    String(priorMeta || '').trim() || '(none)',
    '',
    'Return improved output now.',
  ].join('\n');
}

// Load .env (no dotenv dependency needed)
const envFile = path.join(__dirname, '.env');
if (fs.existsSync(envFile)) {
  fs.readFileSync(envFile, 'utf8').split('\n').forEach(line => {
    const m = line.match(/^\s*([^#=\s][^=]*?)\s*=\s*(.*?)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
  });
}

// ── Parse CLI args ────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
function arg(name) {
  const i = argv.indexOf('--' + name);
  return i !== -1 ? argv[i + 1] : undefined;
}

// Load config file if given (flags override it)
let cfg = {};
const cfgFile = arg('config') || 'run.config.json';
if (fs.existsSync(cfgFile)) {
  cfg = JSON.parse(fs.readFileSync(cfgFile, 'utf8'));
  console.log(`[config] Loaded ${cfgFile}`);
}

const desc        = arg('desc')     || cfg.desc;
const stack       = arg('stack')    || cfg.stack     || 'Python';
const mode        = arg('mode')     || cfg.mode      || 'code';
const nc          = parseInt(arg('nc') || cfg.nc     || 4);
const outFile     = arg('out')      || cfg.out       || 'output/generated.txt';
const requestedArch     = arg('arch')     || cfg.arch      || '';
const requestedReviewer = arg('reviewer') || cfg.reviewer  || '';
const fileCtx     = arg('context')  || cfg.fileCtx   || '';
const instructions= arg('instr')    || cfg.instructions || '';
const usePolish     = cfg.usePolish   || false;
const polishModel   = cfg.polishModel || ''; // OpenRouter model ID for polish pass

// ctxFiles: array of file paths to read and inject as context (for patch/refactor modes)
const ctxFilePaths = Array.isArray(cfg.ctxFiles) ? cfg.ctxFiles : [];
let autoFileCtx = fileCtx;
if (!autoFileCtx && ctxFilePaths.length) {
  autoFileCtx = ctxFilePaths.map(fp => {
    const abs = path.isAbsolute(fp) ? fp : path.join(path.dirname(path.resolve(cfgFile)), fp);
    try {
      const src = fs.readFileSync(abs, 'utf8');
      return `=== ${fp} ===\n${src}`;
    } catch (e) {
      return `=== ${fp} === [ERROR: ${e.message}]`;
    }
  }).join('\n\n');
  if (autoFileCtx) console.log(`[config] Loaded ${ctxFilePaths.length} source file(s) as context`);
}

const ork  = process.env.OPENROUTER_KEY || process.env.OPENROUTER_KEY || cfg.ork;

const requestedAgentModels = Array.isArray(cfg.agents) ? cfg.agents.filter(Boolean) : [];
const skipModels = Array.isArray(cfg.skipModels) ? cfg.skipModels : [];

if (!desc) { console.error('Error: --desc is required (or set "desc" in config file)'); process.exit(1); }
if (!ork)  { console.error('Error: OPENROUTER_KEY env var (or "ork" in config) is required'); process.exit(1); }

// ── Run ID & log setup ────────────────────────────────────────────────────────
const runId   = crypto.randomUUID().slice(0, 8);
const logsDir = path.join(__dirname, 'logs');
const logFile = path.join(logsDir, `run-${runId}.log`);
fs.mkdirSync(logsDir, { recursive: true });
fs.mkdirSync(path.dirname(path.resolve(outFile)), { recursive: true });

function lg(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  fs.appendFileSync(logFile, line + '\n');
}

lg(`Run ${runId} started`);
lg(`desc: ${desc}`);
lg(`stack: ${stack} | mode: ${mode} | agents: ${nc}`);

// ── HTTPS helper ──────────────────────────────────────────────────────────────
async function httpsPost(hostname, urlPath, headers, body) {
  return new Promise((resolve, reject) => {
    let data = '';
    const req = https.request({
      hostname, path: urlPath, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body), ...headers }
    }, res => {
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve({ raw: data, partial: false });
        else reject(new Error(`HTTP ${res.statusCode}: ${data.slice(0, 200)}`));
      });
    });
    const timer = setTimeout(() => {
      req.destroy();
      // Salvage partial response if enough data streamed in
      if (data.length > 200) resolve({ raw: data, partial: true });
      else reject(new Error(`Request timed out after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s`));
    }, REQUEST_TIMEOUT_MS);
    req.on('error', e => { clearTimeout(timer); reject(e); });
    req.on('close', () => clearTimeout(timer));
    req.write(body);
    req.end();
  });
}

async function httpsGet(hostname, urlPath, headers) {
  return new Promise((resolve, reject) => {
    const req = https.request({ hostname, path: urlPath, method: 'GET', headers }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) resolve(data);
        else reject(new Error(`HTTP ${res.statusCode}: ${data.slice(0, 200)}`));
      });
    });
    const timer = setTimeout(() => { req.destroy(); reject(new Error(`Request timed out after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s`)); }, REQUEST_TIMEOUT_MS);
    req.on('error', e => { clearTimeout(timer); reject(e); });
    req.on('close', () => clearTimeout(timer));
    req.end();
  });
}

// Extract content from a potentially truncated OpenRouter JSON response
function extractPartialContent(raw) {
  // Try normal parse first
  try {
    const result = JSON.parse(raw);
    return { content: result.choices?.[0]?.message?.content || '', partial: false };
  } catch {}
  // Try to pull the content string out of incomplete JSON
  const m = raw.match(/"content"\s*:\s*"([\s\S]*)/);
  if (m) {
    // Unescape what we have and strip trailing garbage
    const salvaged = m[1].replace(/\\n/g, '\n').replace(/\\t/g, '\t').replace(/\\"/g, '"').replace(/\\\\/, '\\');
    if (salvaged.length > 100) return { content: salvaged, partial: true };
  }
  return null;
}

async function fetchOpenRouterModelIds() {
  const raw = await httpsGet('openrouter.ai', '/api/v1/models', {
    'Authorization': `Bearer ${ork}`,
    'HTTP-Referer': 'multimodal-agent-cli',
    'X-Title': 'Multimodal Agent CLI'
  });
  const parsed = JSON.parse(raw);
  const models = Array.isArray(parsed?.data) ? parsed.data : [];

  // Cheap paid threshold: $0.0001 per token max (~$100/M tokens)
  // Ling-2.6-flash = $0.00000001/token ✓  |  Claude Sonnet = $0.000003/token ✗ (blocked by prefix anyway)
  const CHEAP_PAID_MAX_PRICE = 0.000001; // $0.000001/token max (~$1/M tokens)

  const isText = id => !['image','audio','lyria','clip','vision','flux'].some(x => id.includes(x));
  const isAllowed = id => !BLOCKED_PROVIDERS.some(p => id.startsWith(p));

  const freeModels = models.filter(m => {
    const id = String(m?.id || '');
    const promptPrice = String(m?.pricing?.prompt || '');
    const isFree = id.endsWith(':free') || promptPrice === '0';
    return isFree && isText(id) && isAllowed(id);
  });

  // Cheap paid fallback — used only when free models are all rate-limited
  // Sorted cheapest first so credits are spent as little as possible
  const cheapPaidModels = models.filter(m => {
    const id = String(m?.id || '');
    const price = parseFloat(m?.pricing?.prompt || '999');
    const isFree = id.endsWith(':free') || price === 0;
    return !isFree && isText(id) && isAllowed(id) && price <= CHEAP_PAID_MAX_PRICE;
  }).sort((a, b) => parseFloat(a?.pricing?.prompt || 0) - parseFloat(b?.pricing?.prompt || 0));

  const sortByPreference = list => list.sort((a, b) => {
    const idA = String(a?.id || ''), idB = String(b?.id || '');
    const rankA = PREFERRED_MODEL_PREFIXES.findIndex(p => idA.startsWith(p));
    const rankB = PREFERRED_MODEL_PREFIXES.findIndex(p => idB.startsWith(p));
    const prefA = rankA === -1 ? 999 : rankA;
    const prefB = rankB === -1 ? 999 : rankB;
    if (prefA !== prefB) return prefA - prefB;
    return Number(b?.context_length || 0) - Number(a?.context_length || 0);
  });

  sortByPreference(freeModels);

  const freeIds = freeModels.map(m => String(m?.id || '').trim()).filter(Boolean);
  const cheapIds = cheapPaidModels.map(m => String(m?.id || '').trim()).filter(Boolean);

  lg(`  Available free models (${freeIds.length}): ${freeIds.slice(0, 6).join(', ')}${freeIds.length > 6 ? '...' : ''}`);
  if (cheapIds.length) lg(`  Cheap paid fallback (${cheapIds.length}): ${cheapIds.slice(0, 4).join(', ')} [used only if free exhausted]`);

  // Warn if known reliable fallbacks are gone
  const RELIABLE_FREE = ['baidu/cobuddy:free', 'openrouter/owl-alpha'];
  const gone = RELIABLE_FREE.filter(id => !freeIds.includes(id));
  if (gone.length) lg(`  [warn] Reliable fallback models no longer free: ${gone.join(', ')}`);
  if (freeIds.length < 3 && !cheapIds.length) lg(`  [warn] Very few models available — runs may fail`);

  // Free models first, cheap paid appended as last-resort fallback
  return [...new Set([...freeIds, ...cheapIds])];
}

function resolveModelSelection(requested, availableIds, label) {
  const candidates = availableIds.filter(id => !isModelCoolingDown(id));
  if (!candidates.length) throw new Error('No available models returned by OpenRouter.');
  if (requested) {
    if (candidates.includes(requested)) return requested;
    lg(`  [warn] Requested ${label} model unavailable: ${requested}. Using live first available model.`);
  }
  return candidates[0];
}

function isModelCoolingDown(modelId) {
  const until = modelCooldownUntil.get(modelId) || 0;
  return Date.now() < until;
}

function markModelCooldown(modelId, reason = '') {
  modelCooldownUntil.set(modelId, Date.now() + MODEL_COOLDOWN_MS);
  lg(`  [cooldown] ${modelId} for ${Math.round(MODEL_COOLDOWN_MS / 60000)}m${reason ? ` (${reason})` : ''}`);
}

function withPhaseTimeout(promise, timeoutMs, phaseLabel) {
  let timer = null;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${phaseLabel} timed out after ${Math.round(timeoutMs / 1000)}s`)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

// ── OpenRouter call ───────────────────────────────────────────────────────────
// Returns { content, partial } — partial=true means the response was cut off mid-stream
async function orCall(modelId, sys, usr, attempt = 0) {
  const body = JSON.stringify({
    model: modelId,
    max_tokens: 4096, // keep responses focused; continuation handles longer outputs
    temperature: 0.3, // lower temp = more deterministic, fewer hallucinations
    messages: [{ role: 'system', content: sys }, { role: 'user', content: usr }]
  });
  let resp;
  try {
    resp = await httpsPost('openrouter.ai', '/api/v1/chat/completions', {
      'Authorization': `Bearer ${ork}`,
      'HTTP-Referer': 'multimodal-agent-cli',
      'X-Title': 'Multimodal Agent CLI'
    }, body);
  } catch (e) {
    const lower = String(e.message || '').toLowerCase();
    if (lower.includes('429') || lower.includes('timed out') || lower.includes('404') || lower.includes('no endpoints')) {
      markModelCooldown(modelId, e.message);
    }
    if (attempt < MAX_TIMEOUT_RETRIES && (e.message.includes('429') || e.message.includes('timed out'))) {
      const wait = Math.pow(2, attempt + 1) * 1000 + Math.random() * 500;
      lg(`  ↻ ${modelId} retry ${attempt + 1}/${MAX_TIMEOUT_RETRIES} in ${Math.round(wait / 1000)}s (${e.message})`);
      await new Promise(r => setTimeout(r, wait));
      return orCall(modelId, sys, usr, attempt + 1);
    }
    throw e;
  }
  if (resp.partial) {
    lg(`  [partial] ${modelId} timed out mid-stream — salvaging ${resp.raw.length} bytes`);
    const extracted = extractPartialContent(resp.raw);
    if (extracted) {
      lg(`  [partial] Salvaged ${extracted.content.length} chars of content`);
      return extracted; // { content, partial: true }
    }
    throw new Error(`Request timed out after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s (no salvageable content)`);
  }
  const result = JSON.parse(resp.raw);
  return { content: result.choices?.[0]?.message?.content || '(empty)', partial: false };
}


// ── Continuation & helpers ────────────────────────────────────────────────────
function looksComplete(text) {
  if (!text || text.length < 80) return false;
  return /[}\]);'"`]$/.test(text.trimEnd());
}

async function withContinuation(modelId, sys, usr, priorPartial = null) {
  // If we have salvaged partial output from a previous model, complete it instead of starting fresh
  if (priorPartial) {
    lg(`  [salvage] Completing partial output (${priorPartial.length} chars) on ${modelId}`);
    const tail = priorPartial.slice(-1200);
    const cont = await orCall(modelId,
      'You are completing code that was cut off mid-generation. Output ONLY the missing remainder — do not repeat anything already written. Complete all functions, close all braces, finish the file.',
      `The code was cut off here. Continue from exactly this point:\n\n${tail}`);
    return priorPartial + cont.content;
  }
  const resp = await orCall(modelId, sys, usr);
  let text = resp.content;
  // If this call itself came back partial, return as-is so the agent loop can pass it to the next model
  if (resp.partial) return text;
  for (let i = 0; i < MAX_CONTINUATION_PASSES; i++) {
    if (looksComplete(text)) break;
    lg(`  ↻ Truncated — continuation attempt ${i + 1}/${MAX_CONTINUATION_PASSES}`);
    const tail = text.slice(-800);
    const cont = await orCall(modelId,
      'You are continuing your previous response. Output ONLY the continuation — do not repeat anything already written. Complete the entire remaining output.',
      `Continue exactly from here (do not repeat):\n${tail}`);
    if (!cont.content || cont.content.trim().length < 10) break;
    text += cont.content;
  }
  return text;
}

function extractJSON(raw) {
  let s = raw.replace(/```json|```/g, '').trim();
  const m = s.match(/\[[\s\S]*\]/);
  if (m) s = m[0];
  return JSON.parse(s);
}

// ── System prompt maps ────────────────────────────────────────────────────────
const archSysMap = {
  report:   `Split task into exactly ${nc} non-overlapping modules. Return JSON array only: module, spec, outputFile.`,
  docs:     `Split task into exactly ${nc} non-overlapping modules. Return JSON array only: module, spec, outputFile.`,
  patch:    `Split task into exactly ${nc} non-overlapping modules. Return JSON array only: module, spec, outputFile.`,
  refactor: `Split task into exactly ${nc} non-overlapping modules. Return JSON array only: module, spec, outputFile.`,
  debug:    `Split task into exactly ${nc} non-overlapping modules. Return JSON array only: module, spec, outputFile.`,
  test:     `Split task into exactly ${nc} non-overlapping modules. Return JSON array only: module, spec, outputFile.`,
  migrate:  `Split task into exactly ${nc} non-overlapping modules. Return JSON array only: module, spec, outputFile.`,
  code:     `Split task into exactly ${nc} non-overlapping modules. Return JSON array only: module, spec, outputFile.`,
};

const agentSysMap = {
  report:   `Write concise markdown for this module only. Return markdown only.`,
  docs:     `Write concise markdown for this module only. Return markdown only.`,
  patch:    `Write ONLY the changes specified for this module. Use the exact language and stack specified (e.g. Node.js, not Python). Follow any output format instructions precisely (unified git diff, code snippet, etc.). No explanations, no markdown fences unless instructed. Surgical edits only.`,
  refactor: `Implement this module and return complete code only.`,
  debug:    `Implement this module and return complete code only.`,
  test:     `Implement this module and return complete code only.`,
  migrate:  `Implement this module and return complete code only.`,
  code:     `Implement this module and return complete code only.`,
};

const revSysMap = {
  report:   `Merge modules into concise final markdown only.`,
  docs:     `Merge modules into concise final markdown only.`,
  patch:    `Merge all module patches into one final unified output. Preserve file headers and hunk markers. No markdown fences, no explanations — output only.`,
  refactor: `Merge modules into one complete final code output only.`,
  debug:    `Merge modules into one complete final code output only.`,
  test:     `Merge modules into one complete final code output only.`,
  migrate:  `Merge modules into one complete final code output only.`,
  code:     `Merge modules into one complete final code output only.`,
};

// ── Main pipeline ─────────────────────────────────────────────────────────────
async function run() {
  const availableModelIds = await fetchOpenRouterModelIds();
  if (!availableModelIds.length) throw new Error('OpenRouter returned no models.');

  // Hard block: never allow Claude or OpenAI to run as agents (burn credits instantly)
  const filteredModelIds = availableModelIds.filter(id =>
    !skipModels.includes(id) &&
    !BLOCKED_PROVIDERS.some(p => id.startsWith(p))
  );
  const arch = resolveModelSelection(requestedArch, filteredModelIds, 'architect');
  const reviewer = resolveModelSelection(requestedReviewer || arch, filteredModelIds, 'reviewer');
  const liveAvailableIds = filteredModelIds.filter(id => !isModelCoolingDown(id));
  // Preferred agents: first nc models (round-robin assignment). Fallback pool = ALL available models.
  const preferredAgentIds = requestedAgentModels.length
    ? requestedAgentModels.filter(id => liveAvailableIds.includes(id))
    : liveAvailableIds.slice(0, Math.max(nc, 1));
  if (!preferredAgentIds.length && !filteredModelIds.length) {
    throw new Error('No agent models available after live availability filtering.');
  }
  // Full fallback pool — all models usable when preferred are rate-limited
  const allAgentIds = filteredModelIds;
  const agentModels = (preferredAgentIds.length ? preferredAgentIds : filteredModelIds.slice(0, nc))
    .map(id => ({ id, label: id.split('/').pop() || id }));
  lg(`Resolved models dynamically from OpenRouter: arch=${arch}, reviewer=${reviewer}, agents=${agentModels.map(a => a.id).join(', ')} (+${allAgentIds.length - agentModels.length} fallbacks)`);

  const isReportMode = mode === 'report';
  const isDocMode    = mode === 'docs';
  const archSys = archSysMap[mode] || archSysMap.code;
  const baseTask = (isReportMode || isDocMode)
    ? `Task: ${desc}\nDomain: ${stack}`
    : `Project: ${desc}\nStack: ${stack}`;
  const protocolTask = buildLearningProtocolPrompt(baseTask, instructions);
  const archUsr = `${protocolTask}${autoFileCtx ? '\n\nContext:\n' + autoFileCtx : ''}`;

  // ── Phase 1: Architect (with multi-model fallback) ─────────────────────────
  // Build ordered list: requested arch first, then remaining available models
  const archCandidates = [arch, ...filteredModelIds.filter(id => id !== arch && !isModelCoolingDown(id))];
  let mods;
  let archUsed = null;
  for (const archModel of archCandidates) {
    if (isModelCoolingDown(archModel)) continue;
    lg(`[1/4] Architect (${archModel})`);
    try {
      const raw = (await withPhaseTimeout(orCall(archModel, archSys, archUsr), ARCHITECT_TIMEOUT_MS, 'Architect phase')).content;
      let parsed;
      try {
        parsed = extractJSON(raw);
      } catch {
        lg('  JSON malformed — retrying with fix prompt...');
        const fixRaw = (await withPhaseTimeout(orCall(archModel,
          'Return ONLY a valid JSON array — no markdown, no explanation, no preamble. Fix any syntax errors.',
          `This JSON is broken, fix it and return ONLY the corrected array:\n${raw}`), ARCHITECT_TIMEOUT_MS, 'Architect JSON fix')).content;
        parsed = extractJSON(fixRaw);
      }
      if (!Array.isArray(parsed)) throw new Error('Architect did not return an array');
      mods = parsed;
      archUsed = archModel;
      lg(`  Modules: ${mods.map(m => m.module).join(', ')}`);
      break;
    } catch (e) {
      lg(`  ERROR (${archModel}): ${e.message} — trying next architect model...`);
    }
  }
  if (!mods) {
    lg('  All architect models failed or rate-limited.');
    process.exit(1);
  }

  // ── Phase 2: Parallel agents ────────────────────────────────────────────────
  lg(`[2/4] Launching ${nc} parallel agents...`);
  const agentSys = agentSysMap[mode] || agentSysMap.code;
  const ao = {};

  await Promise.all(mods.map(async (m, i) => {
    const runOneAgent = async () => {
    const moduleTask = (isReportMode || isDocMode)
      ? `Section: ${m.module}\nInstructions: ${m.spec}\nDomain: ${stack}`
      : `Module: ${m.module}\nFile: ${m.outputFile || outFile}\nSpec:\n${m.spec}\nStack: ${stack}`;
    const agentUsr = `${buildLearningProtocolPrompt(moduleTask, instructions)}${autoFileCtx ? '\n\nContext:\n' + autoFileCtx : ''}`;

    // Preferred: round-robin from preferred agents. Fallback: all available models.
    const preferred = agentModels[i % agentModels.length];
    const fallbacks = [preferred, ...allAgentIds
      .filter(id => id !== preferred.id)
      .map(id => ({ id, label: id.split('/').pop() || id }))
    ].filter(x => !isModelCoolingDown(x.id));
    let lastErr = null;
    let salvaged = null; // partial output from a timed-out model

    for (const mdl of fallbacks) {
      try {
        lg(`  Agent ${i + 1} (${mdl.label}): ${m.module}${salvaged ? ' [completing partial]' : ''}`);
        const raw = await withContinuation(mdl.id, agentSys, agentUsr, salvaged);
        // If this model also returned partial, save it for the next fallback
        if (typeof raw === 'string' && raw.length > 100 && !looksComplete(raw)) {
          salvaged = raw;
          lg(`  Agent ${i + 1} (${mdl.label}) partial — passing to next model`);
          continue;
        }
        ao[m.module] = raw.replace(/^```[\w]*\n?/, '').replace(/\n?```$/, '').trim();
        lg(`  Agent ${i + 1} ✓`);
        lastErr = null;
        salvaged = null;
        break;
      } catch (e) {
        lastErr = e;
        markModelCooldown(mdl.id, e.message);
        const retryable = e.message.includes('429') || e.message.includes('timed out') || e.message.includes('No endpoints');
        lg(`  Agent ${i + 1} (${mdl.label}) failed: ${e.message}${retryable ? ' — trying next model' : ''}`);
        if (!retryable) break;
      }
    }
    // If we ran out of models but have salvaged partial output, use it
    if (lastErr && salvaged) {
      lg(`  Agent ${i + 1}: using best partial output (${salvaged.length} chars)`);
      ao[m.module] = salvaged;
    } else if (lastErr) {
      lg(`  Agent ${i + 1}: all models exhausted — ${lastErr.message}`);
      ao[m.module] = `# ERROR: ${lastErr.message}`;
    }
    };
    try {
      await withPhaseTimeout(runOneAgent(), AGENT_TIMEOUT_MS, `Agent ${i + 1} phase`);
    } catch (e) {
      ao[m.module] = `# ERROR: ${e.message}`;
      lg(`  Agent ${i + 1} timed out: ${e.message}`);
    }
  }));

  // ── Phase 3: Reviewer ───────────────────────────────────────────────────────
  lg(`[3/4] Reviewer (${reviewer})`);
  const revSys  = revSysMap[mode] || revSysMap.code;
  const revIntro = (isReportMode || isDocMode) ? 'Consolidate into a final report:\n\n' : 'Integrate all modules into one complete output:\n\n';
  const revTask = buildLearningProtocolPrompt(
    `${revIntro}Apply the protocol and produce the strongest integrated output.`
  );
  const revUsr  = revTask + '\n\n' + Object.entries(ao).map(([k, v]) => `### ${k}\n${v}`).join('\n\n---\n\n');
  let integrated;
  try {
    integrated = (await withPhaseTimeout(orCall(reviewer, revSys, revUsr), REVIEWER_TIMEOUT_MS, 'Reviewer phase')).content;
    let loops = 1;
    const lowSignal = !integrated || integrated.trim().length < 80 || integrated.includes('NO_VALID_DIFF');
    if (lowSignal && loops < MAX_LEARNING_LOOPS) {
      lg('  Low-signal reviewer output — running one refinement pass...');
      const refineUsr = buildRefinementPrompt(revTask, integrated, 'Reviewer phase low-signal output');
      const refined = (await withPhaseTimeout(orCall(reviewer, revSys, refineUsr), REVIEWER_TIMEOUT_MS, 'Reviewer refinement')).content;
      if (refined && refined.trim().length > (integrated || '').trim().length) integrated = refined;
      loops++;
    }
    lg('  Review complete ✓');
  } catch (e) {
    lg(`  Reviewer failed (${e.message}) — concatenating modules`);
    integrated = Object.entries(ao).map(([k, v]) => `# === ${k} ===\n${v}`).join('\n\n');
  }

  // ── Phase 4: Polish (OpenRouter — best available free model) ──────────────
  let final = integrated;
  if (usePolish) {
    const polisherId = polishModel
      || availableModelIds.find(id => !isModelCoolingDown(id) && id !== reviewer)
      || reviewer;
    lg(`[4/4] Polish (${polisherId})`);
    const polishSys = (isReportMode || isDocMode)
      ? 'You are a senior technical editor. Fix errors, unclear writing, broken structure. Return ONLY corrected markdown.'
      : 'You are a senior engineer. Fix bugs, missing imports, broken syntax, security issues, and placeholder stubs in this AI-generated code. Return ONLY the corrected complete code.';
    try {
      const polished = await withPhaseTimeout(
        orCall(polisherId, polishSys, `Fix and return corrected output:\n\n${integrated}`),
        60000, 'Polish phase'
      );
      const polishedText = polished.content || polished;
      if (polishedText && polishedText.trim().length > 80) {
        final = polishedText.replace(/^```[\w]*\n?/, '').replace(/\n?```$/, '').trim();
        lg('  Polish complete ✓');
      } else {
        lg('  Polish returned empty — using reviewer output');
      }
    } catch (e) {
      lg(`  Polish failed (${e.message}) — using reviewer output`);
    }
  }

  // Deterministic post-filter: keep docs/report outputs concise when requested.
  if ((mode === 'docs' || mode === 'report') && wantsConciseOutput(desc)) {
    final = enforceConciseMarkdown(final);
    lg(`  Applied concise post-filter (${DEFAULT_CONCISE_CHAR_LIMIT} chars max)`);
    const requestedLines = extractRequestedLineCount(desc);
    if (requestedLines) {
      final = enforceExactLineCount(final, requestedLines);
      lg(`  Applied exact line-count post-filter (${requestedLines} lines)`);
    }
  }

  // Strip code fences from non-doc/report output
  if (mode !== 'docs' && mode !== 'report') {
    final = final.replace(/^```[\w]*\r?\n?/, '').replace(/\r?\n?```\s*$/, '').trim();
  }

  // ── Write output ────────────────────────────────────────────────────────────
  const absOut = path.resolve(outFile);
  fs.mkdirSync(path.dirname(absOut), { recursive: true });
  fs.writeFileSync(absOut, final);
  lg(`Output written → ${absOut}`);
  lg(`Log → ${logFile}`);
  lg('Done.');
}

run().catch(e => { lg(`FATAL: ${e.message}`); process.exit(1); });







