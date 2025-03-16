/** Memory mode: discrete GPU or unified memory. */
export type MemoryMode = 'DISCRETE_GPU' | 'UNIFIED_MEMORY';

/** Model quantization set: F32, F16, Q8, Q6, Q5, Q4, Q3, Q2, GPTQ, AWQ. */
export type ModelQuantization =
  | 'F32'
  | 'F16'
  | 'Q8'
  | 'Q6'
  | 'Q5'
  | 'Q4'
  | 'Q3'
  | 'Q2'
  | 'GPTQ'
  | 'AWQ';

/** KV cache quantization: F32, F16, Q8, Q5, Q4. */
export type KvCacheQuantization = 'F32' | 'F16' | 'Q8' | 'Q5' | 'Q4';

/** Recommendation for final output. */
export interface Recommendation {
  gpuType: string;         // e.g., 'Single 24GB GPU' or 'Unified memory...'
  vramNeeded: string;      // e.g., "32.5"
  fitsUnified: boolean;    // relevant if memoryMode = 'UNIFIED_MEMORY'
  systemRamNeeded: number; // in GB
  gpusRequired: number;    // discrete GPUs required (0 if doesn't fit)
}