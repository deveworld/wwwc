# AGENTS.md — TriStore-BMA 프로젝트 컨텍스트

이 파일은 AI 에이전트가 이 프로젝트에 참여할 때 빠르게 컨텍스트를 잡을 수 있도록 작성되었다.

---

## 1. 프로젝트 정체성

**이름:** TriStore-BMA (Budgeted Memory Allocation)
**제출 목표:** ICLR 2026 main track
**핵심 정체성:** budgeted test-time memory allocation paper
**레포:** https://github.com/deveworld/wwwc

### 한 문장 요약

> 고정된 추가 추론 예산(extra end-to-end latency) 아래에서, long-context 입력의 어떤 정보를 parametric write(LoRA adapter update)에, 어떤 정보를 exact cache(verbatim span retention)에 배분해야 하는지, 그리고 hybrid allocation이 single-store specialization보다 언제 더 좋은 accuracy-efficiency Pareto frontier를 만드는지를 연구한다.

### 이 논문이 아닌 것

- 새로운 아키텍처 논문이 아니다 (frozen pretrained LM 위에서 작동)
- SOTA를 전면 돌파하는 논문이 아니다 (internal baseline 대비 Pareto 개선을 보인다)
- Pure RAG 논문이 아니다 (retrieval은 수단이고, allocation law가 contribution)
- Pure TTT 논문이 아니다 (write는 한 축이고, cache와의 budget trade-off가 핵심)

---

## 2. 연구 아이디어 상세

### 2.1 문제 정의

Long-context 입력을 처리할 때, 모델이 모든 토큰을 한 번에 볼 수 없거나 비효율적인 상황에서, 추가 예산을 두 가지 substrate에 배분한다:

1. **Write (parametric adaptation):** frozen LM 위의 LoRA adapter를 chunk-level self-supervised loss로 inference-time update. 정보를 모델 파라미터에 "쓴다."
2. **Cache (exact retention):** raw document에서 선택한 span을 final prompt에 verbatim으로 materialize. 정보를 정확히 "보존한다."

이 두 substrate는 서로 다른 failure mode를 해결한다:
- **Write가 강한 곳:** diffuse dependency (multi-hop reasoning, 정보가 여러 chunk에 분산)
- **Cache가 강한 곳:** exact recall (특정 이름, 코드, 숫자 등의 정확한 재현)

### 2.2 왜 이것이 새로운가

기존 문헌은 write/update 쪽(qTTT, GDWM, In-Place TTT)과 exact retention 쪽(SR-TTT, WG-KV)을 각각 따로 발전시켰다. 누구도 **같은 frozen LM 위에서, 같은 extra-latency budget으로, write와 cache를 interleave하는 allocation을** 정식화하지 않았다.

### 2.3 핵심 가설

- **H1:** exact-recall slice에서는 cache-only > write-only
- **H2:** dependency-heavy slice에서는 write-only > cache-only
- **H3:** mixed-failure slice에서는 hybrid > single-store (핵심 claim)
- **H4:** optimal split은 task mixture에 따라 이동
- **H5:** hybrid 이득은 "더 많은 compute" 때문이 아니라 "예산을 나눠 쓴 결과"

### 2.4 Method: TriStore-BMA Pipeline

```
raw document → chunk → preselector(top-K) → signal pass(surprisal)
                                                    ↓
                                          marginal-utility allocator
                                           ↙                ↘
                                    write(LoRA)        cache(exact spans)
                                           ↘                ↙
                               stable scaffold + cache + query → decode
                                    (+ write adapter state)
```

1. Raw document를 chunk로 나눔
2. Cheap preselector가 top-K candidate chunk 선별
3. Shortlisted chunks에 대해 scaffold-prefixed signal pass (chunk loss 측정)
4. Marginal-utility interleaving allocator: 매 step마다 write marginal gain vs cache marginal gain 비교, 더 큰 쪽에 1 unit 배분
5. Final prompt = [instruction] + [query] + [scaffold] + [cache spans], write state는 adapter로 반영

### 2.5 Baseline 전략 (internal only)

| Baseline | Write | Cache | 설명 |
|----------|-------|-------|------|
| Stable-only | off | off | scaffold만 사용 |
| Write-only | on | off | write에만 예산 집중 |
| Cache-only | off | on | cache에만 예산 집중 |
| Hybrid | on | on | interleaving allocation |
| Thinking | — | — | Gemma 4 native thinking mode |
| Random | random | random | 동일 예산, 랜덤 배분 |
| Oracle | oracle | oracle | synthetic에서 정답 기반 upper bound |

External SOTA(qTTT, GDWM 등)는 regime/substrate가 달라 main에서 직접 비교하지 않음. Appendix에서 faithful reproduction 시도.

### 2.6 Main Metric

**Budget-AUC:** Pareto frontier 아래 면적. 단일 budget point 승패보다 안정적.

보조: low/medium/high budget score, category score, oracle gap closure.

### 2.7 Benchmark

- **RULER:** synthetic, failure mode 분해 (retrieval, multi-hop, aggregation, QA)
- **LongBench v2:** 현실형, 503 MCQ, 6 category, 8k-2M words
- **ZeroSCROLLS:** support/appendix only

Mixed slice는 preregistered (결과와 무관하게 고정):
- retrieval + tracing
- retrieval + aggregation
- tracing + QA

---

## 3. 현재 상황 (2026-04-03 기준)

### 3.1 완료된 것

#### CPU Scaffolding (100%)
- 전체 pipeline CPU 시뮬레이션: chunking, scaffold, preselector, cache, allocator
- CLI 10+ subcommands (calibrate, inspect-manifest, validate-manifest, cross-validate, budget-check, generate-matrix, prompt-smoke, runner-smoke, env-snapshot, summarize-artifacts)
- Preregistered manifests (ruler_mixed_slices, longbench_v2_selection)
- 18 tests passing, ruff/pyright/ty clean
- Sample datasets (ruler_mini.jsonl, longbench_v2_mini.jsonl)

#### GPU Bring-Up (완료)
- **Colab T4 (15.6GB):** Gemma 4 E2B-it 로딩, chat template 검증, thinking/non-thinking 출력 확인
- **A100 MIG 2g-20GB:** Go/No-Go 4개 체크 전부 PASS

#### Verified Technical Specs
- **Model ID:** `google/gemma-4-E2B-it` (대문자 E2B, gated model)
- **API:** `AutoProcessor` (not AutoTokenizer) — Gemma 4는 multimodal
- **Message format:** `[{"type": "text", "text": "..."}]` (multimodal content format)
- **Thinking tags:** `<|channel>thought ... <|/channel>`
- **Dependencies:** pillow, torchvision 필수 (text-only에서도)
- **CUDA 12.x:** torch 2.5.x+cu121 사용 (2.11은 CUDA 13 필요)

#### TransformersGemmaRunner (구현 완료)
- 실제 모델 로딩, inference, thinking/non-thinking generation
- `parse_thinking_response()`: thought/answer 분리
- `VARIANT_TO_HF_ID` mapping: E2B, E4B, A4B, D31B

### 3.2 Go/No-Go 결과 (2026-04-03, A100 MIG)

| Check | 결과 | 데이터 |
|-------|------|--------|
| Scaffold quality | PASS | 3개 substrate 정확히 언급, 2.1s |
| Write vs Stable | PASS | Stable→"Carol", Write→"Alice,Bob,Carol,Dave,Eve" |
| Cache exact recall | PASS | No-cache→hallucination, Cache→exact code |
| Thinking mode | PASS | thought tag 정상, 12x 느림 |

**VERDICT: GO**

### 3.3 아직 완료되지 않은 것

- **LoRA write-step update 구현:** adapter inference-time training 미구현 (가장 중요한 미완성)
- **실제 write branch 검증:** "LoRA 1-step update가 adapter state만으로 accuracy를 올리는가?" — 미검증
- **Hybrid interior split 검증:** allocator가 non-trivial split을 생성하는가? — 미검증
- **Route overhead 측정 및 budget grid 확정**
- **Full RULER/LongBench v2 데이터 로더**
- **E4B bring-up** (main paper default)

### 3.4 핵심 학술적 리스크 (솔직한 평가)

1. **Write branch 작동 여부 (치명적):** LoRA 1-step update가 5B 모델에서 downstream accuracy를 올리지 못하면, hybrid story 자체가 "잘 설계된 RAG"로 퇴화함. 현재 go/no-go check 2는 LoRA가 아니라 prompt enrichment를 테스트한 것임.

2. **Toy theory의 약함:** expected error 분해는 기초 볼록최적화 수준. ICLR 리뷰어가 "trivially obvious"라고 공격할 가능성 높음.

3. **External baseline 부재:** internal baseline만으로 "시스템 안에서 hybrid가 낫다"는 보이지만, "시스템 자체가 의미 있다"는 보이기 어려움. Full-context baseline 최소 하나 필요.

4. **모델 규모:** E4B (5B급)가 메인. 2026 ICLR 기준으로 작을 수 있음. 26B A4B에서도 같은 패턴이 보여야 generalizability 확보.

---

## 4. 계획

### Phase 1: RTX PRO 6000 Calibration (Week 1-2)

**목표:** write branch 작동 여부 확인 + budget grid 확정

1. Gemma 4 E4B bring-up on RTX PRO 6000
2. **LoRA write-step 구현** — PEFT/LoRA adapter, 1-step gradient update per chunk
3. **Write branch 진짜 검증:**
   - dependency-heavy RULER slice에서 write-only (LoRA update) vs stable-only
   - 이것이 실패하면 write objective 수정 또는 write를 auxiliary로 축소
4. Scaffold 강도 calibration (near-floor도 near-ceiling도 아닌 범위 확인)
5. Route overhead 측정 → budget grid 확정
6. K, chunk size, span length 확정

**Week 2 종료 시 잠글 것:** variant, chat template, scaffold policy, K, chunk size, span length, budget grid, mixed slice definitions

### Phase 2: Synthetic Closure (Week 3-6)

**목표:** RULER에서 core claim 검증

1. RULER main sweep (niah, variable_tracking, aggregation, QA)
2. Oracle upper bound 계산
3. K sensitivity study
4. Route overhead study
5. Write-cache tradeoff curves
6. **Core claim 판정:** mixed slice에서 hybrid > single-store?

**Kill switch:** Week 6에서 core claim이 안 보이면 LongBench v2로 진행하지 않음.

### Phase 3: Realistic Evaluation (Week 7-12)

**목표:** LongBench v2에서 real-world generalization

1. LongBench v2 main categories (multi-doc QA, code repo, structured data, dialogue)
2. Raw-length bucket reporting
3. Budget-AUC 계산
4. Mixed-slice analysis

### Phase 4: Support & Writing (Week 8-16)

**목표:** 논문 완성

1. Week 8부터 병렬 집필 (Intro, Related Work, Formulation, Theory, Methods figure)
2. ZeroSCROLLS support (appendix)
3. External baseline appendix (faithful reproduction 가능 시)
4. 2D Pareto, ablation matrix
5. Full-context baseline 추가 (main text)

### 논문 구조 (9 pages)

| Section | Pages | 내용 |
|---------|-------|------|
| Intro | 0.9 | 연구 질문, 핵심 결과 요약 |
| Related Work | 0.6 | write/cache/architecture/evaluation 4축 |
| Problem Formulation | 0.9 | 3 substrate, allocation 변수, budget 제약 |
| Method | 1.4 | pipeline, scaffold, signal, write, cache, allocator |
| Theory | 0.5 | small falsifiable proposition (축소 권장) |
| Experimental Setup | 0.8 | benchmarks, baselines, metrics |
| Main Results | 2.2 | Budget-AUC, Pareto, failure mode analysis |
| Analysis/Ablations | 1.1 | K sensitivity, split dynamics, oracle gap |
| Limitations | 0.6 | reproducibility, ethics |

---

## 5. 코드베이스 구조

```
wwwc/
├── research_plan.md          # 잠긴 연구 계획 v5.1
├── pyproject.toml            # uv 기반 Python 프로젝트
├── configs/
│   ├── calibration_cpu.yaml  # CPU smoke test용
│   ├── gpu_handoff.yaml      # 첫 GPU 실행용 (E4B)
│   └── simulation_cpu.yaml   # LongBench v2 시뮬레이션용
├── manifests/
│   ├── ruler_mixed_slices.yaml       # preregistered RULER slices
│   ├── longbench_v2_selection.yaml   # preregistered LongBench v2
│   └── experiment_registry.yaml      # 전체 실험 레지스트리
├── data/samples/             # CPU 검증용 미니 데이터셋
├── src/tristore_bma/
│   ├── config.py             # GemmaVariant enum, config dataclasses
│   ├── chunking.py           # sliding window chunking
│   ├── scaffold.py           # TF-IDF + lexical + coverage penalty
│   ├── preselector.py        # top-K shortlisting
│   ├── cache.py              # rarity-based span selection
│   ├── allocator.py          # marginal-utility interleaving
│   ├── simulation.py         # toy error model
│   ├── pipeline.py           # end-to-end CPU pipeline
│   ├── gemma_runner.py       # CpuEchoGemmaRunner + TransformersGemmaRunner
│   ├── prompting.py          # multimodal message format
│   ├── logging_utils.py      # RunAccounting schema
│   ├── budgeting.py          # budget calibration rules
│   ├── matrix.py             # run matrix generator
│   ├── datasets.py           # JSONL loader + filter
│   ├── manifests.py          # YAML manifest loader
│   ├── validation.py         # manifest + dataset validation
│   ├── reporting.py          # artifact aggregation
│   ├── repro.py              # environment snapshot
│   └── cli.py                # 10+ subcommands
├── tests/                    # 18 tests (all passing)
├── scripts/
│   ├── load_gemma.py         # Colab bring-up script
│   └── go_nogo.py            # Go/No-Go validation (4 checks)
├── docs/
│   ├── GPU_HANDOFF.md        # GPU 실행 체크리스트 + 잠긴 사양
│   └── BRINGUP_LOG.md        # 시간순 bring-up 기록
└── artifacts/                # 실행 결과물
```

### 핵심 파일별 역할

| 파일 | 수정 빈도 | 역할 |
|------|-----------|------|
| `gemma_runner.py` | 높음 | LoRA write-step 구현 시 수정 필요 |
| `prompting.py` | 낮음 | multimodal message format 잠금 완료 |
| `allocator.py` | 중간 | real signal 기반 allocation으로 전환 시 |
| `pipeline.py` | 높음 | GPU pipeline 통합 시 대폭 수정 |
| `config.py` | 낮음 | variant/schema 안정화됨 |
| `cli.py` | 중간 | 새 subcommand 추가 시 |

---

## 6. 환경 및 인프라

### 개발 환경 (로컬)
- Linux, Python 3.14, uv 기반
- CPU-only (GPU 없음)
- ruff + pyright + ty + pytest로 CI

### GPU 환경

| 환경 | GPU | VRAM | 용도 | torch |
|------|-----|------|------|-------|
| Colab | T4 | 15.6GB | 초기 bring-up | 2.11.0 |
| Elice Cloud | A100 MIG 2g-20GB | 20.9GB | Go/No-Go | 2.5.1+cu121 |
| RTX PRO 6000 | RTX PRO 6000 | 96GB | 본 실험 | TBD |

### 주의사항
- CUDA 12.x 환경에서는 `torch==2.5.x+cu121` 사용 (최신 torch는 CUDA 13 필요)
- `requires-python >= 3.10` (3.10 환경 호환)
- HuggingFace gated model: `HF_TOKEN` 환경변수 또는 `huggingface-cli login` 필요
- `google/gemma-4-E2B-it` 모델 페이지에서 라이선스 동의 필수

---

## 7. 작업 시 주의사항

### 코드 수정 시
- `src/` 수정 후 반드시 `uv run pytest` 실행
- `uvx ruff check src/ tests/` 와 `uvx pyright src/ tests/` 통과 확인
- Gemma 4 message format은 반드시 multimodal: `[{"type": "text", "text": "..."}]`
- `enable_thinking`은 `processor.apply_chat_template()`에 전달, 메시지에 넣지 않음

### 실험 원칙 (research_plan.md §14에서)
- Benchmark마다 scaffold를 다르게 튜닝하지 않는다
- 결과를 보고 mixed slice를 다시 정의하지 않는다
- Budget grid를 결과를 보고 바꾸지 않는다
- Main text에서 external baselines에 과도하게 기대지 않는다
- 모든 run에서 wall-clock latency, GPU-seconds, route overhead, K, seed 등을 로그로 남긴다

### 논문 톤
- "existing methods fail" → "existing methods often address one failure mode at a time"
- "our method consistently outperforms" → "the hybrid frontier dominates on mixed-failure slices"
- "novel architecture" → 절대 사용 금지
- "not RAG" → 사용 금지 (RAG-like 요소를 인정하되 contribution 위치를 명확히)
