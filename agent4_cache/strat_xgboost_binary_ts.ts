$ cat /opt/ai-trader/src/strategies/xgboost_binary_ts.ts
--- rc=0 ---
/**
 * XGBoost Binary Classifier — pure-TypeScript inference.
 *
 * Loads XGBoost 3.x `save_model(raw_format="json")` files for binary:logistic
 * models and runs `predict_proba(features)` without Python at runtime.
 *
 * Model JSON format (XGBoost 3.x native, dumped via `booster.save_model(...)`):
 * ```json
 * {
 *   "learner": {
 *     "learner_model_param": { "base_score": "5E-1", "boost_from_average": "1", ... },
 *     "objective": { "name": "binary:logistic", ... },
 *     "gradient_booster": {
 *       "name": "gbtree",
 *       "model": {
 *         "gbtree_model_param": { "num_trees": "150", "num_parallel_tree": "1" },
 *         "trees": [
 *           {
 *             "base_weights":     [0.0116, 1.16, -1.05, 1.42, ..., -0.0538],
 *             "left_children":    [1, 3, 5, 7, 9, 11, 13, -1, -1, -1, ...],
 *             "right_children":   [2, 4, 6, 8, 10, 12, 14, -1, -1, -1, ...],
 *             "split_indices":    [17, 17, 22, 29, 29, 28, 17, 0, 0, ...],
 *             "split_conditions": [0.9997, 0.9993, 0.000667, 20.0, ...],
 *             "default_left":     [0, 0, 0, 0, ...],
 *             "tree_param": { "num_nodes": "15", "num_feature": "31", ... }
 *           },
 *           ... (150 trees, max_depth=3 → 15 nodes each)
 *         ]
 *       }
 *     }
 *   },
 *   "version": ["3", "0", "0"]
 * }
 * ```
 *
 * Inference (binary:logistic):
 *   1. For each tree, walk from node 0 until reaching a leaf (-1 in left_children).
 *      - At each internal node: if feature[split_indices[node]] < split_conditions[node] → go to left_children[node], else → right_children[node].
 *      - NaN / undefined feature → follow `default_left[node]` (1 = left, 0 = right). Falls back to going LEFT if the field is absent (matches the legacy v2 meta_selector.ts behaviour and the task spec's "default to 0, go left").
 *   2. Sum the base_weights[leaf_node] across ALL trees.
 *   3. Add base_score (converted from probability to logit; default 0.5 → logit 0 → no effect).
 *   4. Apply sigmoid: P = 1 / (1 + exp(-logit)).
 *
 * Verified to match Python xgboost 2.1.3 `predict_proba(...)[1]` within <1e-4
 * on 10 random test vectors for `regime_range_tight.json`.
 */
import * as fs from 'fs'
import * as path from 'path'

// ─── Types ────────────────────────────────────────────────────────────────────

interface XGBoostTree {
  id: number
  base_weights: number[]      // length = num_nodes; leaf value (also populated for internal nodes — unused there)
  left_children: number[]     // -1 means leaf
  right_children: number[]
  split_indices: number[]     // feature index per node (0 for leaves)
  split_conditions: number[] // threshold per node
  default_left?: number[]     // 1 = NaN goes left, 0 = NaN goes right. Absent → default left.
  tree_param: { num_deleted?: string; num_nodes?: string; num_feature?: string; size_leaf_vector?: string }
}

interface XGBoostBinaryModelJson {
  learner: {
    learner_model_param?: {
      base_score?: string          // e.g. "5E-1" or "0.5" (probability for binary:logistic)
      boost_from_average?: string
      num_feature?: string
      num_class?: string
      num_target?: string
    }
    objective?: { name?: string; reg_loss_param?: { scale_pos_weight?: string } }
    gradient_booster: {
      name: string                  // "gbtree"
      model: {
        gbtree_model_param?: { num_trees?: string; num_parallel_tree?: string }
        trees: XGBoostTree[]
        tree_info?: number[]
        iteration_indptr?: number[]
      }
    }
    attributes?: { best_iteration?: string; best_score?: string }
    feature_names?: string[] | number[]
    feature_types?: string[]
  }
  version?: string[]
}

/**
 * Compiled, runtime-ready model. Trees are kept verbatim (fast walk on each
 * prediction); baseMargin is the logit of `base_score` (0 when base_score=0.5).
 */
export interface XGBoostBinaryModel {
  nTrees: number
  nFeatures: number
  trees: XGBoostTree[]
  baseMargin: number               // logit(base_score); 0 for the default 0.5
  baseScoreProb: number           // raw base_score probability (informational)
  objective: string                // "binary:logistic"
  scalePosWeight: number          // 1 if not specified
  sourcePath: string
}

// ─── Model cache ─────────────────────────────────────────────────────────────

const _modelCache = new Map<string, XGBoostBinaryModel>()

// ─── Parsing helpers ──────────────────────────────────────────────────────────

function parseNum(v: string | undefined, def: number): number {
  if (v === undefined || v === null) return def
  const n = Number(v)
  return Number.isFinite(n) ? n : def
}

/**
 * Convert a base_score probability to a logit (margin).
 * For binary:logistic, XGBoost stores `base_score` in probability space
 * (0.5 by default). The actual margin added to the leaf sum is logit(p).
 * logit(0.5) = 0, so the default contributes nothing.
 *
 * NOTE: When `boost_from_average=1` AND `scale_pos_weight` is set, XGBoost
 * currently does NOT update base_score from the training average (it stays
 * at the input default of 0.5). This matches all v4 regime models. If future
 * models actually update base_score (e.g. to 0.27), this conversion is still
 * correct.
 */
function probToLogit(p: number): number {
  if (!Number.isFinite(p)) return 0
  // Clamp to (1e-7, 1 - 1e-7) to avoid ±Infinity
  const clamped = Math.min(1 - 1e-7, Math.max(1e-7, p))
  return Math.log(clamped / (1 - clamped))
}

// ─── Tree walk ───────────────────────────────────────────────────────────────

/**
 * Walk a single XGBoost tree from its root (node 0) and return the leaf value.
 *
 * - Internal node: left_children[node] !== -1
 * - Leaf node:     left_children[node] === -1  (return base_weights[node])
 *
 * Direction logic (XGBoost semantics):
 *   val < threshold   → left
 *   val >= threshold  → right
 *   val is NaN/undef  → default_left[node] === 1 ? left : right
 *                       (if default_left absent → left, per task spec)
 */
function evalTree(tree: XGBoostTree, features: number[]): number {
  const left = tree.left_children
  const right = tree.right_children
  const splitIdx = tree.split_indices
  const splitCond = tree.split_conditions
  const baseWeights = tree.base_weights
  const defaultLeft = tree.default_left
  const n = left.length

  let node = 0
  // max_depth is 3 for v4 models → at most 4 hops. Use 64 as a hard guard.
  for (let depth = 0; depth < 64; depth++) {
    if (left[node] === -1) {
      // Leaf
      return baseWeights[node]
    }
    const featureIdx = splitIdx[node]
    const threshold = splitCond[node]
    const val = features[featureIdx]

    let goLeft: boolean
    if (val === undefined || Number.isNaN(val)) {
      goLeft = defaultLeft ? defaultLeft[node] === 1 : true
    } else {
      goLeft = val < threshold
    }
    node = goLeft ? left[node] : right[node]
    if (node < 0 || node >= n) {
      // Defensive: corrupt index — bail out with current node's leaf weight
      return baseWeights[node] ?? 0
    }
  }
  // Depth exhausted (should never happen with sane max_depth)
  return baseWeights[node] ?? 0
}

// ─── Sigmoid ─────────────────────────────────────────────────────────────────

/**
 * Numerically-stable sigmoid: 1 / (1 + exp(-x)).
 * Clamps the argument to [-50, 50] to avoid Math.exp overflow; outside that
 * range the result is indistinguishable from 0 or 1 in float64 anyway.
 */
export function sigmoid(x: number): number {
  if (x >= 50) return 1
  if (x <= -50) return 0
  if (x >= 0) {
    const z = Math.exp(-x)
    return 1 / (1 + z)
  }
  const z = Math.exp(x)
  return z / (1 + z)
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Load an XGBoost binary model from a JSON file. Results are cached by
 * absolute path; subsequent calls with the same path return the cached
 * instance without re-reading the file.
 *
 * @param modelPath Absolute path to a `regime_*.json` file (XGBoost 3.x raw_format=json)
 */
export function loadModel(modelPath: string): XGBoostBinaryModel {
  const abs = path.isAbsolute(modelPath) ? modelPath : path.resolve(process.cwd(), modelPath)

  const cached = _modelCache.get(abs)
  if (cached) return cached

  if (!fs.existsSync(abs)) {
    throw new Error(`XGBoostBinary: model file not found: ${abs}`)
  }

  const raw = fs.readFileSync(abs, 'utf-8')
  let json: XGBoostBinaryModelJson
  try {
    json = JSON.parse(raw)
  } catch (e: any) {
    throw new Error(`XGBoostBinary: failed to parse JSON at ${abs}: ${e.message}`)
  }

  const learner = json.learner
  if (!learner || !learner.gradient_booster || !learner.gradient_booster.model) {
    throw new Error(`XGBoostBinary: missing learner.gradient_booster.model in ${abs}`)
  }

  const trees = learner.gradient_booster.model.trees
  if (!Array.isArray(trees) || trees.length === 0) {
    throw new Error(`XGBoostBinary: no trees found in ${abs}`)
  }

  const learnerModelParam = learner.learner_model_param || {}
  const baseScoreProb = parseNum(learnerModelParam.base_score, 0.5)
  const baseMargin = probToLogit(baseScoreProb)

  const numFeature = parseNum(learnerModelParam.num_feature, 0)
  // Fallback: derive nFeatures from split_indices (max feature seen) if num_feature is missing.
  let nFeatures = numFeature > 0 ? numFeature : 0
  if (nFeatures === 0) {
    let maxIdx = 0
    for (const t of trees) {
      for (const idx of t.split_indices) if (idx > maxIdx) maxIdx = idx
    }
    nFeatures = maxIdx + 1
  }

  const objective = learner.objective?.name ?? 'binary:logistic'
  const scalePosWeight = parseNum(
    learner.objective?.reg_loss_param?.scale_pos_weight,
    1,
  )

  const model: XGBoostBinaryModel = {
    nTrees: trees.length,
    nFeatures,
    trees,
    baseMargin,
    baseScoreProb,
    objective,
    scalePosWeight,
    sourcePath: abs,
  }

  _modelCache.set(abs, model)
  return model
}

/**
 * Clear the model cache (useful for tests / hot-reload).
 */
export function clearModelCache(): void {
  _modelCache.clear()
}

/**
 * Run inference on a single feature vector.
 *
 * Returns P(positive class) ∈ (0, 1). For v4 regime models the positive
 * class = "price moves up >0.1% within 30 min" (per metadata threshold=0.001,
 * horizon_minutes=30).
 *
 * @param model   Loaded model (from `loadModel`)
 * @param features Feature vector in the SAME ORDER as `feature_names` in the
 *                 training metadata (31 features for v4 regime models).
 */
export function predict_proba(model: XGBoostBinaryModel, features: number[]): number {
  if (!model || !model.trees || model.trees.length === 0) {
    throw new Error('XGBoostBinary.predict_proba: model has no trees')
  }
  if (!Array.isArray(features)) {
    throw new TypeError('XGBoostBinary.predict_proba: features must be an array of numbers')
  }

  // Sum leaf values across all trees + base_margin.
  let logit = model.baseMargin
  for (let t = 0; t < model.trees.length; t++) {
    logit += evalTree(model.trees[t], features)
  }

  return sigmoid(logit)
}

/**
 * Convenience wrapper — accepts a model path (loads+caches) and a feature
 * vector, returns P(positive class).
 */
export function predictProbaFromPath(modelPath: string, features: number[]): number {
  return predict_proba(loadModel(modelPath), features)
}

/**
 * Convenience: return the binary trading decision implied by a probability,
 * using the v4 metadata thresholds:
 *   P > 0.6  → +1 (LONG)
 *   P < 0.4  → -1 (SHORT)
 *   else     →  0 (FLAT)
 *
 * Thresholds can be overridden (e.g. for per-regime tuning).
 */
export function decisionFromProba(
  p: number,
  longThreshold = 0.6,
  shortThreshold = 0.4,
): -1 | 0 | 1 {
  if (!Number.isFinite(p)) return 0
  if (p > longThreshold) return 1
  if (p < shortThreshold) return -1
  return 0
}

// ─── Self-test (run with: bun /home/z/my-project/xgboost_binary_ts.ts) ──────────

// Detect "run as a script". Works under Bun (`import.meta.main`) and Node/tsx
// (`require.main === module`).
const isMain =
  (typeof import.meta === 'object' &&
    (import.meta as any).main === true) ||
  (typeof require !== 'undefined' &&
    typeof require.main !== 'undefined' &&
    require.main === module)

if (isMain) {
  const args = process.argv.slice(2)
  const modelPath = args[0] || '/home/z/my-project/regime_range_tight_sample.json'

  console.log(`=== XGBoost Binary TS — self-test ===`)
  console.log(`Model: ${modelPath}`)

  const model = loadModel(modelPath)
  console.log(`  nTrees: ${model.nTrees}`)
  console.log(`  nFeatures: ${model.nFeatures}`)
  console.log(`  objective: ${model.objective}`)
  console.log(`  baseScoreProb: ${model.baseScoreProb}`)
  console.log(`  baseMargin (logit): ${model.baseMargin}`)
  console.log(`  scalePosWeight: ${model.scalePosWeight}`)

  // Smoke test 1: 31 zeros
  const zeros = new Array(model.nFeatures).fill(0)
  let p0 = predict_proba(model, zeros)
  console.log(`\n  predict(zeros)  → P = ${p0.toFixed(6)}  (decision=${decisionFromProba(p0)})`)

  // Smoke test 2: NaN features
  const nans = new Array(model.nFeatures).fill(NaN)
  let pNan = predict_proba(model, nans)
  console.log(`  predict(NaNs)   → P = ${pNan.toFixed(6)}  (decision=${decisionFromProba(pNan)})`)

  // Smoke test 3: random features (seeded for reproducibility)
  // Mulberry32 PRNG → deterministic output
  let seed = 1337
  const rand = () => {
    seed = (seed + 0x6D2B79F5) | 0
    let t = seed
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const randFeatures = Array.from({ length: model.nFeatures }, () => (rand() - 0.5) * 2)
  const pRand = predict_proba(model, randFeatures)
  console.log(`  predict(random) → P = ${pRand.toFixed(6)}  (decision=${decisionFromProba(pRand)})`)
  console.log(`  random features sample: [${randFeatures.slice(0, 5).map(v => v.toFixed(3)).join(', ')}, ...]`)

  // Smoke test 4: verify against Python test vectors if available
  const vectorsPath = '/home/z/my-project/regime_range_tight_test_vectors.json'
  if (fs.existsSync(vectorsPath)) {
    try {
      const tv = JSON.parse(fs.readFileSync(vectorsPath, 'utf-8'))
      if (Array.isArray(tv.test_inputs) && Array.isArray(tv.expected_proba)) {
        console.log(`\n  Verifying against Python xgboost predictions (${tv.test_inputs.length} samples)...`)
        let maxDiff = 0
        let maxIdx = 0
        let nOk = 0
        for (let i = 0; i < tv.test_inputs.length; i++) {
          const feats = tv.test_inputs[i]
          const expected = tv.expected_proba[i]
          const got = predict_proba(model, feats)
          const diff = Math.abs(got - expected)
          if (diff > maxDiff) { maxDiff = diff; maxIdx = i }
          if (diff < 1e-4) nOk++
          console.log(`    sample ${i}: TS=${got.toFixed(6)}  PY=${expected.toFixed(6)}  Δ=${diff.toExponential(2)}`)
        }
        console.log(`\n  → max diff: ${maxDiff.toExponential(3)} (sample ${maxIdx})`)
        console.log(`  → ${nOk}/${tv.test_inputs.length} within 1e-4`)
        if (maxDiff < 1e-4) {
          console.log(`  ✅ TS inference matches Python xgboost!`)
        } else if (maxDiff < 1e-2) {
          console.log(`  ⚠️  Small drift (<1e-2) — likely float precision; check base_score handling if problematic.`)
        } else {
          console.log(`  ❌ Divergence > 1e-2 — investigate base_score / default_left / tree walk.`)
        }
      }
    } catch (e: any) {
      console.log(`  (test vectors verification skipped: ${e.message})`)
    }
  }

  console.log(`\n=== self-test complete ===`)
}


