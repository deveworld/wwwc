# TriStore-BMA 연구 계획 및 실행 로드맵

이 문서는 연구 계획(구 research_plan.md)과 실행 추적(구 TODO.md)을 통합한 단일 소스다.
외부 리뷰 종합(review.md)의 피드백을 반영하여 우선순위와 전략을 재구성했다.

---

## 0. 핵심 연구 질문

> 고정된 추가 end-to-end latency 예산 아래에서, long-context 입력의 어떤 정보를 parametric write(LoRA adapter update)에, 어떤 정보를 exact cache(verbatim span retention)에 배분해야 하는가? 그리고 hybrid allocation이 single-store strategy보다 언제 더 좋은 accuracy-efficiency Pareto frontier를 만드는가?

### Novelty sentence (리뷰 반영, 안전한 버전)

> 같은 frozen pretrained LM 위에서, 같은 extra end-to-end latency budget 아래, LoRA-style parametric write와 verbatim span retention이 서로 경쟁하도록 두고, 어떤 failure mode에서 어느 substrate가 이기며 hybrid가 언제 interior Pareto improvement를 만드는지 실증적으로 밝힌다.

**절대 쓰지 않을 표현:** "first hybrid memory", "novel architecture", "not RAG", "consistently outperforms"

### Publishable claims (리뷰 합의)

- H1/H2 (cache > write on recall, write > cache on dependency): **publishable novelty 아님.** 예비 검증일 뿐.
- **H3:** mixed-failure slice에서 hybrid > single-store. **핵심 claim. 논문의 존재 이유.**
- **H4:** optimal split이 task mixture에 따라 이동. **interior split figure.**
- **H5:** 이득이 "compute 더 써서"가 아니라 "allocation을 잘해서" 나오며, **confound 통제 필수.**

### Intellectual center = Allocator

논문 서사의 중심은 **"marginal-utility interleaving allocator가 언제 hybrid advantage를 만드는가"**에 있다. 기존 연구는 substrate 내 allocation (GDWM: write 내, Ada-KV: cache 내). TriStore는 substrate 간 allocation.

---

## 0A. 논문 포지셔닝

### 제외 범위
- 새로운 recurrent / hybrid backbone 논문
- large-scale from-scratch pretraining 논문
- pure RAG 논문
- pure KV-compression 논문
- pure TTT 개선 논문
- 모든 최신 long-context baseline을 정면으로 이기는 SOTA 논문

### 포함 범위
- frozen pretrained LM 위의 **test-time allocation law** 논문
- write vs exact cache의 trade-off를 **같은 예산 축**으로 회계하는 논문
- mixed-failure long-context tasks에서 **hybrid allocation이 필요한 이유**를 empirical로 보이는 논문
- bounded materialized prompt를 사용하지만, contribution은 retrieval이 아니라 **allocation principle**

### RAG-like 구조에 대한 stance
이 프로젝트는 bounded final prompt를 만들기 위해 stable scaffold와 selected cache spans를 materialize한다. 따라서 **구조적으로 RAG-like하다는 공격이 온다.** 이 점은 본문에서 먼저 인정한다.

차이는 분명히 적는다:
- 기존 RAG의 핵심 공헌: retrieval-augmented generation 자체
- 이 논문의 핵심 공헌: 고정된 추가 latency 예산 아래, exact retention과 parametric adaptation 사이의 **allocation law**
- retrieval/scaffold는 stable always-on substrate의 구현 수단이며, 새로운 claim의 중심은 아님

---

## 0B. 정식 문제 정의

### 세 substrate

1. **Stable scaffold** S
   - always-on. raw document 전체에서 cheap retrieval/extraction으로 만든 bounded support context
   - 모든 baseline과 method가 **동일하게 공유** (fairness)
   - 구성: query lexical overlap + cheap embedding similarity + document coverage rule
   - fixed number of scaffold spans / fixed total scaffold token budget

2. **Write state** W
   - frozen base model 위의 small fast-weight adapter state (LoRA/adapters)
   - chunk-level self-supervised NTP loss로 inference-time update
   - write는 prompt에 materialize되지 않고 **adapter state로만** 반영

3. **Exact cache** C
   - raw document에서 골라낸 exact spans/tokens
   - 최종 prompt에 **명시적으로 materialize** (verbatim)

### Allocation 변수

raw document를 m개 chunk C_1, ..., C_m으로 나눈다.

- write allocation: 각 chunk에 대해 write step 수 k_i ∈ Z≥0
- cache allocation: 각 후보 span s에 대해 binary keep variable m_s ∈ {0,1}

### 예산 제약

메인 budget axis는 **extra end-to-end latency**.

```
B = τ_method - τ_stable
```

τ_stable = stable-only pipeline latency, τ_method = 임의 method latency.

같은 hardware, 같은 software stack, batch size 1, 동일 decode setting에서 측정.

secondary logging (별도 기록):
- extra GPU-seconds
- total write steps
- total cached tokens/spans
- route/signal overhead
- decode overhead

부록에서는 compute vs cache-size 2D Pareto도 제공.

### Raw document length vs Materialized prompt length

- **raw document length:** allocator가 볼 수 있는 전체 원문 길이
- **materialized prompt length:** 실제 base LM이 마지막 decode 시 보는 prompt 길이

이 구분이 중요한 이유:
- internal methods (hybrid/write/cache)는 bounded materialized prompt regime에서 동작
- full-context baselines는 실제 prompt 길이가 모델 context limit에 직접 묶임

이 차이를 논문 전체에서 명시해야 E2B/E4B와 32k/64k/128k의 혼선이 사라진다.

### Provisional budget grid

```
B ∈ {0, 0.5, 1.0, 1.5} × τ_stable
```

Week 1-2 calibration 후 확정. Calibration decision rules:
- median route cost > medium budget의 40%: K 줄이거나 grid 확대
- median route cost > small budget의 80%: small-B 제거

---

## 0C. Method 상세: TriStore-BMA Pipeline

### High-level pipeline

```
1. raw document → chunk (sliding window, overlap)
2. cheap preselector → 전체 문서에서 상위 K개 candidate chunk 선별
3. shortlisted chunks에 대해 scaffold-prefixed signal pass (loss/surprisal 측정)
4. marginal-utility interleaving allocator → write에 줄지 cache에 줄지 결정
5. stable scaffold + cached spans + query → materialize, write adapter state 적용 → decode
```

### Signal pass

각 shortlisted chunk C_i에 대해:

```
[scaffold prefix] + [C_i]
```

를 모델에 넣고, **loss/surprisal는 오직 C_i 부분에서만** 계산.

이렇게 하면 signal의 해석이 "stable scaffold로도 충분히 커버되지 않는 residual difficulty"가 되어 story와 맞는다.

chunk를 단독으로 넣으면 시작 부분 loss가 인위적으로 높아지는 artifact 발생하므로, scaffold-prefixed가 기본.

### Write branch

- frozen base LM + small LoRA on selected layers
- 1-step or few-step gradient update per selected chunk
- chunk-wise update only (문서 전체를 한 번에 학습하지 않음)
- Gemma 4 공식 LoRA/QLoRA tuning path 지원

Write utility:
```
U_w(i, r) = L̃_i - δ(r-1)
```
- L̃_i: scaffold-conditioned chunk loss (normalized)
- r: same chunk에 누적 write step index
- δ: repeated write diminishing-return penalty (single scalar)

**핵심 risk:** write는 raw chunk distribution에서 학습하지만 final decode는 scaffold+cache+query distribution. 이 mismatch가 write 효과를 죽일 수 있으므로 Phase 1에서 즉시 검증.

### Cache branch

- signal = surprisal 중심
- exact span length bounded
- top spans materialized verbatim

Cache utility:
```
U_c(s) = S̃(s)
```
- S̃(s): normalized span surprisal score

redundancy penalty는 main에서 제거, appendix ablation으로만.

### Allocator: Marginal-utility interleaving

매 step에서:
1. next write step의 marginal gain = U_w(best available chunk, r+1)
2. next cache span의 marginal gain = U_c(best available span)
3. 더 큰 쪽에 한 단위 예산 배분

write-first greedy를 버린 이유:
- write 편향 내장
- theory의 interior optimum / KKT 해석과 불일치
- interleaving이 "split matters"를 method에서 구현하는 직관적 형태

### Final prompt

```
Prompt = [instruction] + [query] + [scaffold S] + [cached spans C]
```

Write state는 prompt에 materialize되지 않고 adapter state로만 반영.

---

## 0D. Backbone: Gemma 4

### Variant ladder

| Variant | 역할 | 크기 | Context |
|---------|------|------|---------|
| E2B-it | bring-up, robustness, 저비용 재현 | ~5B | 128K |
| **E4B-it** | **main paper default** | ~5B (active) | 128K |
| 26B A4B-it | appendix scale-up | 26B (active ~4B) | 128K |
| 31B dense | 제외 (재현성/비용) | 31B | — |

### 선택 이유
- configurable thinking modes (enable_thinking=True/False)
- native system role
- latest Transformers support
- 공식 LoRA/QLoRA tuning path
- 128K/256K context window

### 운영 원칙
- 한 실험 run 안에서는 오직 하나의 variant만 사용
- stable-only / thinking / write-only / cache-only / hybrid가 항상 동일 variant 공유
- main text 주 결과는 E4B 기준
- E2B는 bring-up / robustness evidence
- 26B A4B는 core claim이 닫힌 뒤 appendix

### Thinking baseline 정의
- main text thinking baseline: Gemma 4 native thinking mode + latency-matched max_new_tokens
- non-thinking baselines: same chat template with enable_thinking=False
- API: `processor.apply_chat_template(..., enable_thinking=True/False)`
- 응답 파싱: `<|channel>thought ... <|/channel>` 태그로 thought/answer 분리

---

## 0E. 관련 문헌 포지셔닝

### Write/update 축
- **qTTT** (ICLR 2026 poster): query-only TTT, thinking보다 gradient update가 효율적. write 쪽 operating point. exact token retention 자체는 주제로 삼지 않음.
- **GDWM** (2026.01): test-time adaptation을 budget-constrained memory consolidation으로 재정식화. 이 논문이 가장 직접적으로 계승하는 문헌. 단, **write 내 allocation만** 다루고 cross-substrate는 다루지 않음.
- **In-Place TTT / LaCT** (ICLR 2026 oral): TTT의 hardware inefficiency 감소, chunk-wise update practical. "write를 하되 반드시 chunk-wise and hardware-aware하게"라는 engineering lesson.
- **PERK** (ICLR 2026 poster): LoRA를 test-time memory module로 사용. 0.5B/7B에서 강한 결과. write substrate의 직접적 선행 연구.

### Exact retention / cache 축
- **SR-TTT** (2026.02): compressed fast-weight memory가 exact recall에서 catastrophic failure. surprisal-driven residual cache로 완화. **가장 직접적 경쟁자.** exact cache branch를 두는 근거이다.
- **WG-KV** (2025.12): long-context inference를 KV admission/selection/eviction으로. trainable gate 기반 KV-centric. parametric write branch와 함께 budget allocation으로 보지는 않음.

### Architecture-level memory 축
- **Titans** (NeurIPS 2025): memory-augmented architecture. "hybrid memory가 필요할 수 있다"는 큰 방향성. 우리와 달리 architecture 자체를 바꿈.
- **ATLAS**: memory-augmented architecture family. frozen LM 위의 inference-time policy가 아니라 architecture 수준.
- **GradMem** (2026.03): memory tokens에 gradient descent로 써 넣음. write의 또 다른 형태.

### 기타 인접
- **TTT-E2E** (2025.12): LoRA continual learning, constant latency. exact recall 실패 인정, cache 필요 근거. **scooping risk**: Sun 그룹이 cache 추가하면 위험.
- **MemOS** (2025.07): plaintext/activation/parameter 3-tier memory. 개념적 중복이지만 budget-allocation 메커니즘 없음.
- **Doc-to-LoRA** (Sakana, 2025): 문서를 LoRA로 내부화. write 문제의식 직접 겹침.

### Evaluation 축
- **RULER**: retrieval + multi-hop tracing + aggregation + QA. failure mode 분해에 가장 깨끗. synthetic.
- **LongBench v2**: 503 MCQ, 8k-2M words, 6 category. 현실형 mixed failure.
- **HELMET**: synthetic NIAH만으로는 downstream 평가 부족하며, RULER + LongBench v2를 함께 쓰는 justification.
- **ZeroSCROLLS**: zero-shot long-text benchmark. support/appendix 역할.
- **SCBench**: shared-context / KV lifecycle / multi-request. main claim 바깥 (single-query regime).

### Testable predictions from theory

toy theory가 만들어야 하는 검증 가능한 예측:
1. mixed ratio가 바뀌면 optimal split이 이동해야 한다 (H4)
2. exact-heavy slice에서는 cache marginal utility가 더 커야 한다
3. dependency-heavy slice에서는 write marginal utility가 더 커야 한다
4. synthetic oracle allocation과 heuristic allocation 사이 gap은 줄어들어야 한다

main에서는 theorem을 크게 밀지 않고, **small but falsifiable proposition**으로 유지. 지면은 allocator dynamics 시각화에 할당.

---

## 1. 완료된 항목

### CPU Scaffolding
- [x] Lock research plan to Gemma 4 family
- [x] Build `uv`-based Python project
- [x] Chunking, scaffold, preselector, cache proposal, allocator simulation
- [x] CLI 10+ subcommands (calibrate, inspect-manifest, validate-manifest, cross-validate, budget-check, generate-matrix, prompt-smoke, runner-smoke, env-snapshot, summarize-artifacts)
- [x] Artifact summary and environment snapshot utilities
- [x] CPU sample datasets and preregistered manifests
- [x] Tests (18 passing), ruff/pyright/ty clean

### GPU Bring-Up
- [x] Model dependencies (`uv sync --extra model --extra dev`)
- [x] `TransformersGemmaRunner` 실제 구현 (AutoProcessor, bfloat16, device_map=auto)
- [x] `apply_chat_template(..., enable_thinking=True/False)` 검증
- [x] Thinking/non-thinking 응답 파싱 (`<|channel>thought ... <|/channel>`)
- [x] Multimodal message format 확정 (`[{"type": "text", "text": "..."}]`)

### Go/No-Go (A100 MIG 2g-20GB, 2026-04-03)
- [x] Scaffold quality: 모델이 scaffold context를 올바르게 사용
- [x] Thinking mode: distinct paths, reproducible
- [x] Prompt enrichment: richer context → better multi-hop (LoRA write가 아닌 prompt 수준)
- [x] Cache recall: exact span → hallucination 제거

### 잠긴 사양
- Model ID: `google/gemma-4-E2B-it` (bring-up), `google/gemma-4-E4B-it` (main)
- API: `AutoProcessor` + `AutoModelForCausalLM`
- Message format: multimodal `[{"type": "text", "text": "..."}]`
- Thinking tags: `<|channel>thought ... <|/channel>`
- Dependencies: pillow, torchvision 필수 (text-only에서도)
- CUDA 12.x: torch 2.5.x+cu121

---

## 2. Phase 1: Write Branch + Calibration (Week 1-2), RTX PRO 6000

**이 Phase가 논문의 생사를 결정한다.**

### 2.1 환경 세팅
- [x] A100 80GB PCIe 환경 세팅 (Elice Cloud, CUDA 12.2, Python 3.10, torch 2.5.1+cu121)
- [x] PEFT/LoRA 라이브러리 통합 (pyproject.toml에 `peft>=0.13.0` 추가)
- [ ] Gemma 4 E4B bring-up (RTX PRO 6000에서)
- [ ] E4B chat template / thinking mode 검증

### 2.1a LoRA Write Fail-Fast (2026-04-03, A100 80GB)

**실험 설계:**
- 모델: `google/gemma-4-E2B-it` (5.12B, bf16)
- LoRA: rank=16, alpha=32, target=q_proj+v_proj, dropout=0
- Write: 1-step SGD (lr=1e-4) on document text (self-supervised NTP loss)
- 5개 dependency-heavy multi-hop 테스트 (chain traversal, cause-effect, inheritance)
- 3가지 조건 비교:
  - **Stable-only:** document in prompt, no LoRA update
  - **Write-only:** document in prompt + 1-step LoRA update on document
  - **Write-nodoc:** 1-step LoRA update on document, then query WITHOUT document in prompt (pure adapter recall)
- Metric: fraction of expected entities found in answer

**판정 기준:**
- write_avg - stable_avg ≥ 0.03: GO
- 0 < delta < 0.03: MARGINAL
- delta ≤ 0: FAIL, pivot

**진행 상태:** 실험 실행 중 (A100 80GB PCIe)

### 2.2 LoRA Write-Step 구현 (P0, acceptance gating)
- [ ] `gemma_runner.py`에 LoRA adapter inference-time update 구현
  - frozen base LM + small LoRA on selected layers
  - chunk-level self-supervised NTP loss
  - 1-step gradient update per selected chunk
  - scaffold-prefixed chunk loss (chunk 시작 artifact 방지)
- [ ] Write-step 비용 측정 (latency, GPU-seconds, memory)

### 2.3 Write Branch Fail-Fast 검증 (P0)
- [ ] **핵심 실험:** RULER dependency-heavy slice에서 write-only > stable-only?
  - variable_tracking task, 32k/64k raw length
  - write-only = scaffold + LoRA adapter update (cache 없음)
  - stable-only = scaffold만
  - 동일 latency budget에서 비교
- [ ] 결과 판정:
  - **성공 (write-only > stable-only by ≥3%):** Phase 2 진행
  - **약한 효과 (0-3%):** Plan B 시도
  - **실패 (write-only ≤ stable-only):** Plan B 시도. 전부 실패 시 pivot

### 2.4 Plan B (write 실패 시)
- [ ] Option A: Query Projection만 타겟팅 (qTTT 방식)
- [ ] Option B: N-step gradient update 허용 (budget 단위를 Δt ms로)
- [ ] Option C: LoRA-GA initialization (gradient-aligned)
- [ ] Option D: MLP Final Layer만 타겟팅 (In-Place TTT 방식)
- [ ] 전부 실패 시: measurement paper / negative-result paper로 pivot

### 2.5 Fixed Split Sweep (P0, interior optimum 확인)
- [ ] Mixed slice에서 write/cache split ratio 변화 실험:
  - 0/100 (cache only)
  - 25/75
  - 50/50
  - 75/25
  - 100/0 (write only)
- [ ] **판정:** interior optimum이 보이면 H3 예비 확인 후 Phase 2 진행. 안 보이면 allocator/signal 재설계.

### 2.6 Budget Calibration
- [ ] Route overhead 실측 (signal pass + preselector 비용)
- [ ] `budget-check` 실행 + budget grid 확정
  - route > small budget 80%: small-B 제거
  - route > medium budget 40%: K 축소 또는 grid 확대
- [ ] 예산 단위 정의 확정: 1 unit = Δt milliseconds. overhead 포함.
- [ ] 다축 회계 프로토콜: token budget, latency, GPU-seconds, route overhead를 같은 표에 기록

### 2.7 Week 2 종료 시 잠글 것
- [ ] Default Gemma 4 variant (E4B)
- [ ] Chat template / response parsing rule
- [ ] Scaffold policy (이후 benchmark별 retuning 금지)
- [ ] Shortlist K
- [ ] Chunk size
- [ ] Cache span length
- [ ] Budget grid
- [ ] Mixed slice definitions (preregistered, 결과와 무관하게 고정)
- [ ] Subset ids / seeds / logging format

---

## 3. Phase 2: Synthetic Closure (Week 3-6)

**Kill switch: core claim (H3)이 안 보이면 LongBench v2로 진행하지 않는다.**

### 3.1 RULER Main Sweep
- [ ] 4개 task: niah (retrieval), variable_tracking (multi-hop), common_words_extraction (aggregation), QA task 1개
- [ ] 3개 raw length bucket: 32k, 64k, 128k
- [ ] 7개 baseline × budget grid:
  - Stable-only, Write-only, Cache-only, Hybrid, Random, Thinking, Oracle
- [ ] 모든 run에서 로그: wall-clock latency, GPU-seconds, route overhead, write steps, cached tokens, materialized prompt length, K, seed, variant

### 3.2 External Baselines (P0, main text에 포함)
- [ ] Full-context baseline: 모델 context limit 내 bucket에서 raw document 전체 입력
- [ ] Naive RAG baseline: top-K chunk를 그대로 prompt에 concat
- [ ] (가능 시) qTTT faithful reproduction, 최소 appendix
- [ ] (가능 시) SnapKV/Ada-KV, cache-only compression baseline

### 3.3 Interior Split Figure (P1, 논문의 심장)
- [ ] X축: write/cache split ratio (또는 budget level), Y축: accuracy
- [ ] Mixed slice에서 interior optimum이 존재함을 보여주는 figure
- [ ] Budget 변화에 따라 optimal split이 이동하는 phase diagram (H4)

### 3.4 Core Ablation (P0)
- [ ] Allocator 제거: 고정 split (50/50) vs learned allocation, allocator 기여 증명
- [ ] Signal pass 제거: random score vs surprisal, signal 가치 증명
- [ ] K-sweep: K=4, 8, 12, 16에서 ranking 안정성
- [ ] Budget grid별 동일 경향 유지 확인

### 3.5 Oracle Upper Bound
- [ ] Synthetic mixed tasks에서 정답 정보 기반 oracle split 계산
- [ ] Oracle gap = oracle - heuristic allocator
- [ ] Oracle-stable gap이 거의 없으면 해당 slice 버림

### 3.6 Route Overhead Study
- [ ] Signal pass 비용의 budget 대비 비율 분석
- [ ] "overhead 포함해도 still beneficial인가?" 에 대한 정량적 답

### 3.7 Week 6 중간 판정 (공식 go/no-go)
- [ ] RULER mixed slice에서 hybrid Budget-AUC > write-only Budget-AUC 인가?
- [ ] RULER mixed slice에서 hybrid Budget-AUC > cache-only Budget-AUC 인가?
- [ ] Interior split figure에서 non-trivial interior optimum이 보이는가?
- [ ] 전부 YES면 Phase 3. 하나라도 NO면 scope 축소 또는 pivot.

---

## 4. Phase 3: Realistic Evaluation (Week 7-12)

### 4.1 LongBench v2 (P1)
- [ ] Main categories: multi-document QA, code repository understanding, long structured data, long dialogue history
- [ ] Raw-length bucket reporting
- [ ] Category별 Budget-AUC
- [ ] Mixed-slice analysis

### 4.2 혼합 실패 Slice 오류 분석 (P1)
- [ ] 어떤 문제에서 write가 실패하고 cache가 구하는지, 반대는 언제인지
- [ ] 정성 분석: 구체적 예시 3-5개
- [ ] 정량 분석: failure mode별 accuracy breakdown

### 4.3 Confound 통제 실험 (H5 방어)
- [ ] Token budget 통제: hybrid와 cache-only가 동일한 materialized prompt length
- [ ] Latency-matched: write-only와 cache-only가 동일한 extra latency
- [ ] 결과: "이득이 compute가 아니라 allocation 덕분"임을 증명

### 4.4 26B A4B Partial Results (P2)
- [ ] RULER 핵심 1-2개를 26B A4B로 실행
- [ ] "Scale-invariant allocation law" 한 문단 작성 (appendix)

### 4.5 Ethics/Privacy (P1)
- [ ] Canary string test: LoRA에 민감정보 잔류 여부
- [ ] Safety alignment 점검: write 후 안전 프롬프트 거부율 변화
- [ ] Ethics statement 작성: privacy, data retention, alignment risk

### 4.6 Statistics
- [ ] Item-level bootstrap confidence interval
- [ ] Paired McNemar test (hybrid vs write-only, hybrid vs cache-only)
- [ ] Multiple comparisons: Holm-Bonferroni

---

## 5. Phase 4: Writing (Week 8-16, 실험과 병렬)

Week 8부터 병렬 집필. 실험 끝나고 쓰는 방식 금지.

### 5.1 논문 구조 (9 pages, 리뷰 반영 수정)

| Section | Pages | 내용 |
|---------|-------|------|
| Intro | 0.9 | 연구 질문, allocation law, 핵심 결과 한 문장 |
| Related Work | 0.7 | write축(qTTT, GDWM, PERK, In-Place TTT), cache축(SR-TTT, WG-KV), architecture축(Titans, ATLAS), evaluation축(RULER, HELMET). **SR-TTT, PERK, GDWM 필수 포함** |
| Problem Formulation | 0.8 | 3 substrate, allocation 변수, budget 제약, materialized vs raw length |
| Method | 1.4 | pipeline, scaffold, signal, write, cache, allocator. Allocator를 전면에 |
| Theory/Proposition | 0.3 | Small falsifiable proposition만. 나머지 삭제 |
| Allocator Dynamics | 0.8 | Split dynamics heatmap, phase diagram, failure mode decomposition (theory 지면 전환) |
| Experimental Setup | 0.8 | benchmarks, baselines (external 포함), metrics, budget accounting |
| Main Results | 2.0 | Budget-AUC, Pareto frontier (external anchor 포함), interior split figure |
| Analysis/Ablations | 0.7 | K sensitivity, allocator ablation, confound control |
| Limitations/Ethics | 0.6 | reproducibility, privacy/safety, alignment risk |

### 5.2 메인 Figures

1. **Figure 1:** Method overview (allocator 중심으로 재구성)
2. **Figure 2:** Pareto frontier on RULER + LongBench v2, **full-context + naive RAG anchor 포함**
3. **Figure 3:** Interior split figure, write/cache ratio vs accuracy on mixed slices (논문의 심장)
4. **Figure 4:** Allocator dynamics heatmap, 어떤 정보가 write/cache에 배분되는지

### 5.3 메인 Tables

1. **Table 1:** Budget-AUC + key budget points (internal + external baselines)
2. **Table 2:** 다축 cost accounting (latency, GPU-seconds, tokens, overhead)

### 5.4 집필 톤 규칙

**쓸 표현:**
- "existing methods often address one failure mode at a time"
- "we study when and how the split between selective write and exact cache matters"
- "the hybrid frontier dominates matched single-store on mixed-failure slices"
- "the allocator is one instantiation; the main finding is that the split matters"

**피할 표현:**
- "existing methods fail"
- "our method consistently outperforms"
- "novel architecture"
- "not RAG" (RAG-like 인정하되 contribution 위치 명확히)
- "first hybrid memory" (SR-TTT, MemOS, GDWM 있음)
- thinking baseline을 main enemy로 두지 않기

### 5.5 Appendix

- 2D Pareto (compute vs cache-size)
- Oracle gap table
- Full ablation matrix
- 26B A4B partial results
- qTTT/GDWM faithful comparison (가능 시)
- Exact subset policy
- Implementation details
- ZeroSCROLLS support (일정 여유 시)

---

## 6. Rebuttal Playbook (리뷰 반영 업데이트)

### "이건 그냥 RAG-like pipeline 아닌가?"
맞다, bounded evidence materialization이 포함된다. 그러나 contribution은 retrieval이 아니라 fixed extra latency budget 아래 exact retention과 parametric adaptation의 allocation law다. Pareto frontier와 interior split figure가 이를 증명한다.

### "compute를 더 쓴 것뿐 아닌가?" (H5)
main budget axis는 measured extra E2E latency이며, route/signal/write/cache/decode overhead를 모두 포함한다. Fixed split 실험에서 동일 budget의 고정 배분보다 learned allocation이 낫다는 것이 H5의 직접적 증거.

### "write branch가 실제로 뭘 하는지 불명확하다."
dependency-heavy slice에서 write-only > stable-only를 보였다. Write-decode distribution mismatch는 scaffold-prefixed loss로 완화했다. Hybrid 이득은 mixed slices에서만 주장한다.

### "prompt 토큰이 늘어서 좋아진 것 아닌가?"
Token budget 통제 실험: hybrid와 cache-only가 동일한 materialized prompt length를 사용하도록 제한한 결과에서도 hybrid가 우위.

### "external SOTA와 왜 직접 비교 안 했나?"
Full-context와 Naive RAG는 main text에 포함했다. qTTT/GDWM은 regime이 달라 faithful reproduction이 appendix에 있다. 이는 claim과 evidence의 alignment을 우선한 선택이다.

### "K나 scaffold를 튜닝해서 이긴 것 아닌가?"
K, scaffold, subset ids, mixed slices는 Week 2에 고정했고, K sensitivity를 main text에 넣었다.

### "5B 모델에서만 됐으면 generalize 되나?"
26B A4B partial results가 appendix에 있으며 동일한 allocation pattern을 보인다.

### "안전/프라이버시 문제는?"
Ethics statement에서 canary test + alignment 점검 결과를 보고했다. Cache는 verbatim span 노출 위험이 있으나 문서 자체의 접근 권한 범위 내에서 동작한다. Write는 adapter state에 정보가 잔류할 수 있으며, inference 종료 시 adapter를 폐기하는 정책을 권장한다.

---

## 7. Subtle Risks & Go/No-Go 체계

### Risk A: Chunk-local forward artifact
- **문제:** chunk 단독 loss 평가 시 시작 부분이 인위적으로 높아짐
- **해결:** scaffold-prefixed chunk loss (기본 고정)
- **Go/No-Go:** scaffold-prefixed와 chunk-only signal이 크게 다르면 chunk-only 폐기

### Risk B: Write-decode distribution mismatch (가장 중요)
- **문제:** write는 raw chunk에서 학습, final decode는 scaffold+cache+query에서 발생
- **Go/No-Go:** dependency-heavy slice에서 write-only가 stable-only를 못 이기면 → Plan B 또는 pivot

### Risk C: Route overhead dominates small budgets
- **해결:** cheap preselector + top-K exact pass + budget recalibration
- **Go/No-Go:** median overhead > small budget 80% → small-B 제거

### Risk D: K sensitivity
- **정책:** K sensitivity는 main analysis inset으로 승격
- **Go/No-Go:** default K 주변에서 ranking 완전 뒤집힘 → preselector 재설계

### Risk E: Scaffold 강도 (too weak / too strong)
- **Go/No-Go:** stable-only가 near-floor → scaffold 확대, near-ceiling → 축소
- Week 2 이후 scaffold policy 동결

### Risk F: Oracle gap
- **Go/No-Go:** oracle-stable gap 거의 없으면 해당 slice 버림. Gap 크지만 hybrid가 못 따라가면 signal/allocator 재설계

### Risk G: Preregistered mixed slice failure
- **정책:** slice 정의 고정. 결과 안 맞으면 "왜 어긋났는지" 분석 자체를 결과로 남김

---

## 8. 경쟁 지형 (Related Work 필수)

### Substrate 간 allocation (novelty 위치)
| 기존 연구 | allocation 범위 | TriStore와의 차이 |
|----------|---------------|-----------------|
| GDWM | write **내** (어떤 chunk에 몇 step) | cross-substrate 안 함 |
| Ada-KV, HeadKV | cache **내** (어떤 KV 유지) | write 없음 |
| qTTT | write 쪽 단일 operating point | allocation 개념 없음 |
| SR-TTT | write + residual cache (surprising tokens) | budget 통합 회계 없음 |
| TriStore-BMA | **substrate 간** (write vs cache 경쟁) | 이것이 novelty |

### 직접 경쟁자 (Related Work 필수)
- **SR-TTT** (2026.02): hybrid memory, 가장 직접적 경쟁자
- **PERK** (ICLR 2026 poster): LoRA as test-time memory module
- **GDWM** (2026.01): budget-constrained memory consolidation
- **TTT-E2E** (2025.12): LoRA continual learning, exact recall 실패
- **qTTT** (ICLR 2026 poster): query-only TTT
- **In-Place TTT** (ICLR 2026 oral): drop-in TTT
- **MemOS** (2025.07): 3-tier memory
- **GradMem** (2026.03): gradient-based memory tokens
- **Doc-to-LoRA** (Sakana, 2025): 문서를 LoRA로 내부화

### Scooping risk
TTT-E2E(Stanford, Yu Sun 그룹)가 selective caching을 추가하는 후속 연구를 내면 TriStore 기여가 크게 줄어들 수 있음. → 속도가 중요.

---

## 9. 벤치마크 & Baseline 사양

### RULER (synthetic, main)
- Tasks: niah, variable_tracking, aggregation, QA
- Raw length: 32k, 64k, 128k
- Preregistered mixed slices: retrieval+tracing, retrieval+aggregation, tracing+QA

### LongBench v2 (realistic, main)
- Categories: multi-document QA, code repository, long structured data, long dialogue
- 503 MCQ, 8k-2M words

### Baselines (main text)

| Baseline | Write | Cache | Budget | 역할 |
|----------|-------|-------|--------|------|
| Stable-only | off | off | 0 | 최소 기준선 |
| Write-only | on | off | matched | dependency 상한 |
| Cache-only | off | on | matched | exact recall 상한 |
| Hybrid | on | on | matched | 핵심 method |
| Random allocation | random | random | matched | allocation 의미 검증 |
| Fixed split (50/50 등) | fixed | fixed | matched | allocator 기여 검증 |
| Thinking | — | — | latency-matched | thinking compute baseline |
| Oracle (synthetic) | oracle | oracle | matched | upper bound |
| **Full-context** | — | — | 무제한 | **external anchor (main text)** |
| **Naive RAG** | — | top-K concat | — | **external anchor (main text)** |

### Main Metric
- **Budget-AUC:** Pareto frontier 아래 면적
- 보조: low/medium/high budget scores, category scores, oracle gap closure

---

## 10. 구현 체크리스트

### 반드시 로그로 남길 것
- wall-clock latency, GPU-seconds
- route overhead (signal pass + preselector)
- write steps per chunk, write latency per step
- cached tokens/spans count
- final materialized prompt length
- raw document length
- scaffold token count
- K, chunk size, span length
- seed, decode config
- thinking on/off, Gemma 4 variant

### 실험 freeze 전에 고정할 것
- Gemma 4 variant, chat template, response parsing
- scaffold construction rule
- K, chunk size, cache span length
- budget grid, mixed slice list
- subset ids, random seeds

### 금지 사항
- benchmark마다 scaffold 다르게 튜닝
- 결과 보고 mixed slice 재정의
- budget grid를 결과 보고 변경
- main text에서 external baselines에 과도 의존

---

## 11. References

### 핵심 문헌
1. ICLR 2026 Author Guide: https://iclr.cc/Conferences/2026/AuthorGuide
2. ICLR 2026 Reviewer Guide: https://iclr.cc/Conferences/2026/ReviewerGuide
3. qTTT: https://openreview.net/forum?id=H0bcEdPCoc
4. RULER: https://github.com/NVIDIA/RULER
5. LongBench v2: https://aclanthology.org/2025.acl-long.183/

### Backbone
6. Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
7. Gemma HF inference: https://ai.google.dev/gemma/docs/core/huggingface_inference
8. Gemma LoRA tuning: https://ai.google.dev/gemma/docs/core/lora_tuning

### 경쟁 문헌
9. SR-TTT: https://arxiv.org/abs/2603.06642
10. GDWM: (2026.01 preprint)
11. TTT-E2E: (2025.12 preprint)
12. PERK: ICLR 2026 poster
13. In-Place TTT: ICLR 2026 oral
14. Titans: https://arxiv.org/abs/2501.00663
15. ATLAS: https://arxiv.org/abs/2505.23735
16. GradMem: https://openreview.net/forum?id=Wdzhnmu5HR
17. WG-KV: https://arxiv.org/abs/2512.17452
18. MemOS: (2025.07)

### 평가
19. ZeroSCROLLS: https://aclanthology.org/2023.findings-emnlp.536/
20. HELMET: https://github.com/princeton-nlp/HELMET
21. SCBench: https://www.microsoft.com/en-us/research/publication/scbench-a-kv-cache-centric-analysis-of-long-context-methods/
