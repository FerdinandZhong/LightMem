# LightMem: Technical Summary

> **Paper**: [LightMem: Lightweight and Efficient Memory-Augmented Generation](https://arxiv.org/abs/2510.18866)
> **Status**: Accepted at ICLR 2026

## Overview

LightMem is a cognition-inspired memory management framework for LLMs that addresses the computational overhead of existing memory systems. It achieves significant efficiency gains while maintaining or improving QA accuracy.

### Key Results

| Metric | Improvement |
|--------|-------------|
| QA Accuracy | +7.7% (LongMemEval) / +29.3% (LoCoMo) |
| Token Reduction | 38x / 20.9x |
| API Call Reduction | 30x / 55.5x |
| Online Test-time Cost | 106x / 117x token reduction |

---

## Architecture: Three-Stage Memory System

Inspired by human cognitive psychology, LightMem implements a hierarchical memory pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT MESSAGES                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SENSORY MEMORY                                                          │
│  ├─ Pre-compression (LLMLingua-2): Token-level content compression      │
│  ├─ Topic Segmentation: Attention-based semantic boundary detection     │
│  └─ Token-aware buffering (default: 512 tokens)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SHORT-TERM MEMORY                                                       │
│  ├─ Segment aggregation from sensory memory                             │
│  ├─ Token threshold tracking (default: 2000 tokens)                     │
│  └─ Triggers extraction when threshold exceeded                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LONG-TERM MEMORY                                                        │
│  ├─ LLM-based fact extraction (metadata generation)                     │
│  ├─ Vector indexing (Qdrant) for semantic retrieval                     │
│  └─ Offline "sleep-time" consolidation updates                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Technical Innovations

### 1. Pre-Compression (Sensory Filtering)

**Purpose**: Rapidly filter irrelevant information through lightweight compression.

**Implementation**: Uses Microsoft's LLMLingua-2 BERT-based model for token-level compression.

```python
# Iterative compression for oversized content
while len(tokenizer.encode(compressed)) >= 512:
    compressed = compressor.compress_prompt(
        context=compressed,
        **compress_config
    )
```

**Key Features**:
- Token-level content compression with configurable rate
- Preserves semantic meaning while reducing token count
- 20-50% token reduction before LLM calls
- Supports both entropy-based and fixed-rate compression

---

### 2. Topic Segmentation (Semantic Grouping)

**Purpose**: Group information by topics for coherent memory organization.

**Two-Stage Algorithm**:

1. **Coarse Stage (Attention-Based)**:
   - Uses transformer layers 8-11 for cross-sentence attention
   - Creates NxN similarity matrix where M[i,j] = attention between sentences
   - Identifies local maxima (topic transition points)

2. **Fine Stage (Similarity-Based)**:
   - Computes cosine similarity between consecutive turns
   - Applies adaptive thresholds (0.2-0.5)
   - Merges boundaries within 3-turn proximity

```
Input: [Turn1, Turn2, Turn3, Turn4, Turn5, Turn6, Turn7]
                    ↓ Attention Analysis
Boundaries: [_, _, _, ↑, _, _, ↑]  (topics at 4, 7)
                    ↓
Output: [[Turn1-3], [Turn4-6], [Turn7]]
```

---

### 3. Memory Entry Structure

Each memory is stored with rich metadata for advanced filtering:

```python
@dataclass
class MemoryEntry:
    id: str                    # UUID
    time_stamp: str            # ISO timestamp
    float_time_stamp: float    # Unix timestamp for range queries
    weekday: str               # Mon, Tue, etc.
    category: str              # Memory category
    memory: str                # Extracted factual content
    original_memory: str       # Original uncompressed
    compressed_memory: str     # Compressed version
    topic_id: int              # Global topic identifier
    topic_summary: str         # Topic-level summary
    speaker_id: str            # Speaker identifier
    hit_time: int              # Access count
    update_queue: List         # Candidates for offline update
    consolidated: bool         # Whether entry was summarized
```

---

### 4. Offline Update ("Sleep-Time" Consolidation)

**Purpose**: Decouple memory consolidation from real-time inference to reduce online costs.

**Two-Phase Process**:

#### Phase 1: Construct Update Queue
```python
for entry in all_entries:
    # Find similar entries with earlier timestamps
    candidates = search(entry.vector, top_k=20, filter="timestamp < entry.timestamp")
    # Store top candidates for later processing
    entry.update_queue = candidates[:keep_top_n]
```

#### Phase 2: Execute Updates
```python
for entry in entries_with_candidates:
    # LLM decides: delete (redundant), update (consolidate), or keep
    action = llm.judge(entry, entry.update_queue)

    if action == "delete":
        remove(entry)  # Entry is redundant
    elif action == "update":
        entry.memory = llm.consolidate(entry, sources)
        entry.consolidated = True
```

**Parallelization**: Uses ThreadPoolExecutor with 8 concurrent workers.

---

### 5. Retrieval Strategies

| Strategy | Backend | Use Case |
|----------|---------|----------|
| **Embedding** | Qdrant (vector) | Semantic similarity search |
| **Context** | BM25 | Keyword/lexical matching |
| **Hybrid** | Both | Combined semantic + keyword |

**Embedding Retrieval Features**:
- Cosine distance similarity
- Metadata filtering (time, weekday, speaker, category)
- Query exclusion (prevent returning source entries)
- Pagination via scroll API

---

## StructMem Extension

StructMem extends LightMem with hierarchical memory for long-horizon reasoning:

### Event-Level Extraction Mode

Goes beyond flat facts to capture:
- **Factual components**: Who, what, when, where
- **Relational components**: Interpersonal dynamics, causal influences
- **Temporal binding**: Event sequences and causality

```python
config = {
    "extraction_mode": "event",  # vs "flat"
}
```

### Cross-Event Summarization

```python
lightmem.summarize(
    retrieval_scope="global",
    time_window=3600,      # seconds
    top_k=15,              # seed memories
    process_all=True
)
```

**Process**:
1. Retrieve entries in time windows
2. Find semantically related seeds via embedding
3. Group seeds by timestamp to reconstruct events
4. Generate cross-event summaries
5. Store summaries in separate collection

---

## Data Flow Pipeline

```
Input Messages (dict/list)
         │
         ▼
┌─────────────────────────────────────┐
│ Message Normalizer                  │
│ - Standardize timestamps            │
│ - Assign unique IDs                 │
│ - Extract weekday info              │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ [Optional] Pre-compression          │
│ - LLMLingua-2 token compression     │
│ - 20-50% token reduction            │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ [Optional] Topic Segmentation       │
│ - Attention-based boundaries        │
│ - Semantic grouping                 │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Sensory → Short-term Buffer         │
│ - Token-aware aggregation           │
│ - Threshold-based extraction        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ LLM Metadata Generation             │
│ - Fact extraction                   │
│ - Category classification           │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ MemoryEntry Creation + Storage      │
│ - Vector embedding                  │
│ - Qdrant indexing                   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ [Offline] Sleep-time Update         │
│ - Construct update queues           │
│ - LLM consolidation                 │
└─────────────────────────────────────┘
```

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `pre_compress` | `False` | Enable LLMLingua-2 compression |
| `topic_segment` | `False` | Enable semantic segmentation |
| `metadata_generate` | `True` | Extract facts from content |
| `text_summary` | `True` | Compress extracted facts |
| `messages_use` | `"user_only"` | Which messages to process |
| `index_strategy` | `"embedding"` | How to index memories |
| `retrieve_strategy` | `"embedding"` | How to retrieve memories |
| `extraction_mode` | `"flat"` | Flat facts or event mode |
| `update` | `"offline"` | When to consolidate |

---

## Supported Backends

| Component | Options |
|-----------|---------|
| **LLM** | OpenAI, DeepSeek, Ollama, vLLM, Transformers |
| **Embeddings** | OpenAI API, HuggingFace (local) |
| **Vector DB** | Qdrant |
| **Pre-compression** | LLMLingua-2, Entropy Compress |
| **Context Retrieval** | BM25 |

---

## Performance Characteristics

| Operation | Optimization |
|-----------|--------------|
| Pre-compression | 20-50% token reduction |
| Topic segmentation | Reduces redundant processing |
| Token-aware buffering | Prevents oversized batches |
| Offline updates | 8 parallel workers |
| Lazy embedding | Only embedded if needed |
| Payload filtering | Reduces search space |

---

## Benchmarks

### LoCoMo Results (gpt-4o-mini backbone)

| Method | Accuracy | Total Tokens (k) | API Calls | Runtime (s) |
|--------|----------|------------------|-----------|-------------|
| FullText | 73.83% | 54,884 | - | 6,971 |
| NaiveRAG | 63.64% | 3,870 | - | 1,884 |
| A-MEM | 64.16% | 21,665 | 11,754 | 67,084 |
| Mem0 | 36.49% | 25,793 | 19,070 | 120,175 |
| **LightMem** | **Best efficiency** | **Lowest** | **Lowest** | **Fastest** |

---

## References

- Paper: https://arxiv.org/abs/2510.18866
- Code: https://github.com/zjunlp/LightMem
- LLMLingua-2: https://github.com/microsoft/LLMLingua
