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

- H1/H2 (cache > write on recall, write > cache on dependency) → **publishable novelty 아님.** 예비 검증일 뿐.
- **H3:** mixed-failure slice에서 hybrid > single-store → **핵심 claim. 논문의 존재 이유.**
- **H4:** optimal split이 task mixture에 따라 이동 → **interior split figure.**
- **H5:** 이득이 "compute 더 써서"가 아니라 "allocation을 잘해서" → **confound 통제 필수.**

### Intellectual center = Allocator

논문 서사는 "write + cache를 섞었다"가 아니라 **"marginal-utility interleaving allocator가 언제 hybrid advantage를 만드는가"**로 서야 한다. 기존 연구는 substrate 내 allocation (GDWM: write 내, Ada-KV: cache 내). TriStore는 substrate 간 allocation.

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

## 2. Phase 1: Write Branch + Calibration (Week 1-2) — RTX PRO 6000

**이 Phase가 논문의 생사를 결정한다.**

### 2.1 환경 세팅
- [ ] RTX PRO 6000 환경 세팅 + Gemma 4 E4B bring-up
- [ ] E4B chat template / thinking mode 검증
- [ ] PEFT/LoRA 라이브러리 통합 (pyproject.toml에 `peft` 추가)

### 2.2 LoRA Write-Step 구현 (P0 — acceptance gating)
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
  - **실패 (write-only ≤ stable-only):** Plan B 시도 → 전부 실패 시 pivot

### 2.4 Plan B (write 실패 시)
- [ ] Option A: Query Projection만 타겟팅 (qTTT 방식)
- [ ] Option B: N-step gradient update 허용 (budget 단위를 Δt ms로)
- [ ] Option C: LoRA-GA initialization (gradient-aligned)
- [ ] Option D: MLP Final Layer만 타겟팅 (In-Place TTT 방식)
- [ ] **전부 실패 시:** measurement paper / negative-result paper로 pivot

### 2.5 Fixed Split Sweep (P0 — interior optimum 확인)
- [ ] Mixed slice에서 write/cache split ratio 변화 실험:
  - 0/100 (cache only)
  - 25/75
  - 50/50
  - 75/25
  - 100/0 (write only)
- [ ] **판정:** interior optimum이 보이면 H3 예비 확인 → Phase 2 진행. 안 보이면 allocator/signal 재설계.

### 2.6 Budget Calibration
- [ ] Route overhead 실측 (signal pass + preselector 비용)
- [ ] `budget-check` 실행 + budget grid 확정
  - route > small budget 80% → small-B 제거
  - route > medium budget 40% → K 축소 또는 grid 확대
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

### 3.2 External Baselines (P0 — main text에 포함)
- [ ] Full-context baseline: 모델 context limit 내 bucket에서 raw document 전체 입력
- [ ] Naive RAG baseline: top-K chunk를 그대로 prompt에 concat
- [ ] (가능 시) qTTT faithful reproduction — 최소 appendix
- [ ] (가능 시) SnapKV/Ada-KV — cache-only compression baseline

### 3.3 Interior Split Figure (P1 — 논문의 심장)
- [ ] X축: write/cache split ratio (또는 budget level), Y축: accuracy
- [ ] Mixed slice에서 interior optimum이 존재함을 보여주는 figure
- [ ] Budget 변화에 따라 optimal split이 이동하는 phase diagram (H4)

### 3.4 Core Ablation (P0)
- [ ] Allocator 제거: 고정 split (50/50) vs learned allocation → allocator 기여 증명
- [ ] Signal pass 제거: random score vs surprisal → signal 가치 증명
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
- [ ] **전부 YES → Phase 3. 하나라도 NO → scope 축소 또는 pivot.**

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
| Method | 1.4 | pipeline, scaffold, signal, write, cache, allocator. **Allocator를 전면에** |
| Theory/Proposition | 0.3 | Small falsifiable proposition만. 나머지 삭제 |
| **Allocator Dynamics** | **0.8** | **Split dynamics heatmap, phase diagram, failure mode decomposition** (theory 지면 전환) |
| Experimental Setup | 0.8 | benchmarks, baselines (external 포함), metrics, budget accounting |
| Main Results | 2.0 | Budget-AUC, **Pareto frontier (external anchor 포함)**, interior split figure |
| Analysis/Ablations | 0.7 | K sensitivity, allocator ablation, confound control |
| Limitations/Ethics | 0.6 | reproducibility, privacy/safety, alignment risk |

### 5.2 메인 Figures

1. **Figure 1:** Method overview (allocator 중심으로 재구성)
2. **Figure 2:** Pareto frontier on RULER + LongBench v2 — **full-context + naive RAG anchor 포함**
3. **Figure 3:** Interior split figure — write/cache ratio vs accuracy on mixed slices (논문의 심장)
4. **Figure 4:** Allocator dynamics heatmap — 어떤 정보가 write/cache에 배분되는지

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
1. ICLR 2026 Author Guide — https://iclr.cc/Conferences/2026/AuthorGuide
2. ICLR 2026 Reviewer Guide — https://iclr.cc/Conferences/2026/ReviewerGuide
3. qTTT — https://openreview.net/forum?id=H0bcEdPCoc
4. RULER — https://github.com/NVIDIA/RULER
5. LongBench v2 — https://aclanthology.org/2025.acl-long.183/

### Backbone
6. Gemma 4 model card — https://ai.google.dev/gemma/docs/core/model_card_4
7. Gemma HF inference — https://ai.google.dev/gemma/docs/core/huggingface_inference
8. Gemma LoRA tuning — https://ai.google.dev/gemma/docs/core/lora_tuning

### 경쟁 문헌
9. SR-TTT — https://arxiv.org/abs/2603.06642
10. GDWM — (2026.01 preprint)
11. TTT-E2E — (2025.12 preprint)
12. PERK — ICLR 2026 poster
13. In-Place TTT — ICLR 2026 oral
14. Titans — https://arxiv.org/abs/2501.00663
15. ATLAS — https://arxiv.org/abs/2505.23735
16. GradMem — https://openreview.net/forum?id=Wdzhnmu5HR
17. WG-KV — https://arxiv.org/abs/2512.17452
18. MemOS — (2025.07)

### 평가
19. ZeroSCROLLS — https://aclanthology.org/2023.findings-emnlp.536/
20. HELMET — https://github.com/princeton-nlp/HELMET
21. SCBench — https://www.microsoft.com/en-us/research/publication/scbench-a-kv-cache-centric-analysis-of-long-context-methods/
