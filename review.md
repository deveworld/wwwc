# TriStore-BMA 외부 리뷰 종합

4개 독립 리뷰와 Codex 종합을 통합한 최종 문서. review 3(에이전트 프레임워크 일반론)은 직접 해당하지 않아 제외.

---

## 1. 전체 판정

모든 리뷰 소스가 동일한 결론에 도달한다.

> **아이디어/프레이밍은 최상급이나, 현재 상태로는 reject. LoRA write branch 실증이 논문의 생사를 가른다.**

| 소스 | 현재 상태 | Write + baseline 해결 시 | 핵심 표현 |
|------|----------|------------------------|----------|
| 리뷰 1 | Borderline~Reject | Spotlight/Oral 가능 | "High Risk, High Return" |
| 리뷰 2 | 20-35% accept | 상한 이동 | "genuine novelty gap, execution-dependent" |
| 리뷰 4 | Low (Lean Reject) | Medium (Poster) | "핵심 기여 미구현" |
| 리뷰 5 | 5-10% (Reject) | 30-45% (Weak Accept) | "좋은 문제 설정, 증거 없음" |
| Codex | "실험 scaffold 수준" | 네 가지 조건 충족 시 경쟁력 | "claim support가 비어 있다" |

---

## 2. 무엇이 publishable하고 무엇이 아닌가

### H1, H2는 publishable novelty가 아니다

- **H1** (exact-recall에서 cache > write): SR-TTT의 residual cache 직관과 이미 맞음.
- **H2** (dependency-heavy에서 write > cache): qTTT/PERK 흐름과 이미 맞음.

이 둘은 "당연히 그렇겠지"로 읽히는 예비 검증이며, 논문의 headline이 될 수 없다.

### H3, H4, H5가 논문의 존재 이유

- **H3:** mixed-failure slice에서 hybrid > single-store → **핵심 claim.**
- **H4:** optimal split이 task mixture에 따라 이동 → **interior split figure의 근거.**
- **H5:** 이득이 "compute 더 써서"가 아니라 "allocation을 잘해서" → **가장 까다로운 방어. confound 통제 필수.**

논문 서사는 H1/H2를 빠르게 확인한 뒤 H3/H4/H5 증명에 지면을 집중해야 한다.

### Allocator가 논문의 intellectual center

리뷰 2의 핵심 통찰:

> Write와 cache 자체는 이미 익숙한 메커니즘이다. 논문에서 가장 distinctive한 intellectual contribution은 **marginal-utility interleaving allocator**다.

- "LoRA + cache를 섞었다"로 쓰면 약하고, "allocator가 언제 hybrid advantage를 만드는가"로 쓰면 강하다.
- 기존 연구와의 구분선:

| 연구 | allocation 범위 | TriStore와의 차이 |
|------|---------------|-----------------|
| GDWM | substrate **내** (write 안에서 어떤 chunk에 몇 step) | cross-substrate 안 함 |
| Ada-KV, HeadKV | substrate **내** (cache 안에서 어떤 KV 유지) | write 없음 |
| qTTT | write 쪽 **단일 operating point** | allocation 개념 없음 |
| TriStore-BMA | substrate **간** (write vs cache 사이에서 어디에) | 이것이 novelty |

---

## 3. 치명적 약점 4개

### 약점 1: Write branch 미검증 (전원 지적, 치명적)

현재 go/no-go check 2는 "더 많은 context를 주면 답이 좋아진다"를 테스트했으며, LoRA parametric write를 테스트하지 않았다.

구체적 비판:
- 리뷰 1: "Write가 안 되면 '복잡하게 꼬아놓은 RAG 논문'으로 전락"
- 리뷰 2: "vanilla LoRA는 full fine-tuning 대비 5-6배 느리게 수렴(LoRA-GA, 2024). 1-step으로는 매우 약한 update만 가능할 수 있다. Dynamic TTA(2026.02)도 'naïve unsupervised TTA with fixed lr can be unstable'이라고 경고"
- 리뷰 4: "이 상태에서 hybrid와 single-store를 비교하면, 사실상 'cache + prompt engineering vs prompt engineering'에 가까워져 핵심 공헌이 붕괴"
- 리뷰 5: "LoRA 1-step이 안정적으로 안 오르면 methods paper로는 버려야 한다"

**Plan B 옵션:**

| 옵션 | 근거 |
|------|------|
| Query Projection만 타겟팅 | qTTT가 이 방식으로 성공 |
| N-step gradient update 허용 | budget 단위를 "1 LoRA step"이 아니라 "Δt ms"로 재정의 |
| LoRA-GA initialization | gradient-aligned init으로 수렴 속도 개선 |
| Adaptive learning rate | Dynamic TTA(2026.02) 참조 |
| MLP Final Layer만 타겟팅 | In-Place TTT 접근 |
| **Pivot** | measurement paper / negative-result paper로 전환 |

### 약점 2: External baseline 부재 (전원 지적, 심각)

- "Regime이 달라 main에서 비교 안 한다"는 논리는 ICLR에서 통하지 않는다.
- ICLR author guide에 따르면 appendix는 리뷰어가 읽을 의무가 없다.
- 리뷰 4: "내부 비교만으로는 시스템 자체의 의미 입증이 약하다."

**main text Pareto 그래프에 반드시 포함:**

| 기준점 | 역할 | 비고 |
|--------|------|------|
| Full-context (가능 길이) | "그냥 다 넣으면?" 상한 anchor | 모델 context limit 내 bucket만 |
| Naive RAG (top-K chunk concat) | cache-only 대비 기준선 | token 길이/비용 통제 |

**main 또는 appendix에 강력 권장:**

| 기준점 | 역할 | 비고 |
|--------|------|------|
| qTTT | "pure write" 기준점 | ICLR 2026 poster, 부재가 눈에 띔 |
| SnapKV 또는 Ada-KV | "pure cache compression" 기준점 | KV-cache 최적화 대표 |
| GDWM | 가장 가까운 경쟁자 | write 내 budget allocation |

리뷰 2의 포지셔닝: "Full-context는 너무 느리고, RAG는 multi-hop에서 무너지지만, Hybrid는 현실적 예산 구간에서 최적 방어선(Pareto frontier)을 형성한다." Hybrid가 무조건 최고일 필요는 없고, 특정 budget 구간에서의 Pareto 우위를 보이면 된다.

### 약점 3: Theory 약함 (전원 지적, 중간)

toy theory(§4.2)의 expected error 분해는 기초 볼록최적화이며, "trivially obvious"라는 공격이 예상된다.

- 리뷰 1: "약한 수학을 길게 쓰면 math-washing으로 읽힌다"
- 리뷰 4: "알고리즘이 무엇을 보장하는지, 가정이 무엇인지가 불명확하면 강하게 reject"

**대안 (택1):**

- **옵션 A (시각화 전환):** Theory ≤ 0.5p. 확보된 지면을 "Allocator Split Dynamics 시각화"에 투자. 어떤 정보(고유명사/코드 = 높은 entropy vs 추상적 맥락)를 만났을 때 allocator가 cache를 택하고 write를 택하는지 heatmap으로 보여줌. (리뷰 1: "시스템 논문에서 리뷰어 극찬을 이끌어내는 치트키")
- **옵션 B (최소 이론 + 측정학):** allocation을 multi-objective knapsack으로 정식화. 단조성/상계 같은 성질 1-2개만 증명. Budget-AUC를 "평가 기여"로 격상. (리뷰 4)
- **핵심 조건:** "hybrid가 dominate한다"만 말하면 불충분. **"어떤 failure-mode mixing 조건에서 hybrid가 dominate하는가"**를 characterize해야 함. (리뷰 2)

### 약점 4: 모델 규모 (3/5 지적, 보통)

E4B(5B) 단독은 generalization 의심을 받을 수 있다.

**최소 방어:** RULER 핵심 1-2개를 26B A4B로 돌려 appendix에 "Scale-invariant allocation law" 한 문단.

**완화 요인:** qTTT(Qwen3-4B), PERK(0.5B/7B), In-Place TTT(4B) 등 ICLR 2026 accepted에서도 5B급 실험은 fatal하지 않다.

---

## 4. 경쟁 위협: research_plan.md에 없는 관련 연구

### 직접적 경쟁자 (Related Work 필수)

| 논문 | 공개일 | 위협 수준 | 핵심 내용 | TriStore와의 관계 |
|------|--------|---------|----------|-----------------|
| **SR-TTT** | 2026.02 | **높음** | highly surprising tokens만 residual cache에 보내는 hybrid memory | 가장 직접적 경쟁자. cache 필요성을 write 쪽에서 인정 |
| **PERK** | ICLR 2026 poster | **높음** | LoRA를 test-time memory module로 사용. 0.5B/7B 강한 결과 | write substrate의 기존 연구 |
| **GDWM** | 2026.01 | **높음** | budget-constrained memory consolidation. CPMI 기반 utility로 write 내 allocation | TriStore가 GDWM을 subsume (cross-substrate로 확장) |
| **TTT-E2E** | 2025.12 | 중간 | LoRA-based continual learning, constant latency. exact recall 실패 인정 | cache 필요 근거. scooping risk: Sun 그룹이 cache 추가하면 위험 |
| **MemOS** | 2025.07 | 중간 | plaintext/activation/parameter 3-tier memory with distillation | 개념적 중복, but budget-allocation 메커니즘 없음 |
| **Latent Context Compilation** | 2026.01 | 중간 | disposable LoRA로 long context compile | write substrate 관련 |
| **GradMem** | 2026.03 | 중간 | memory tokens을 gradient descent로 최적화 | write의 또 다른 형태 |

### 추가 인접 연구 (리뷰 4)

| 논문 | 위협 수준 | 핵심 |
|------|---------|------|
| **NN-TTT** (ICLR 2024) | 중간 | 이웃 문서 retrieval + test-time fine-tune |
| **ActiveFT/SIFT** (ICLR 2025) | 중간 | test-time FT 데이터 선택 |
| **Doc-to-LoRA** (Sakana, 2025) | 중간 | 긴 문서를 LoRA로 내부화, write 문제의식 직접 겹침 |

### 경쟁 지형 3-tier (리뷰 2)

1. **Architectural solutions (배포 규모):** Gated DeltaNet(Qwen3.5), Mamba-2(Nemotron-H), Jamba 1.5. 모델 설계 레벨에서 해결. TriStore의 장점은 "기존 Transformer에 적용 가능"이지만, "임시 패치"로 읽힐 수 있음.
2. **TTT-based methods (직접 경쟁):** qTTT, TTT-E2E, GDWM, In-Place TTT. TriStore의 핵심 차별점: cross-substrate allocation.
3. **KV-cache compression:** SnapKV, Ada-KV, PyramidKV, KVzip 등. TriStore는 이를 "100% cache allocation"의 special case로 subsume.

**가장 큰 scooping risk:** TTT-E2E (Stanford, Yu Sun 그룹)가 selective caching을 추가하는 후속 연구를 내면 TriStore의 기여가 크게 줄어들 수 있음.

### Novelty 방어선 수정

**위험한 문장:** "아무도 hybrid memory를 안 했다" → SR-TTT, MemOS, GDWM이 있어 즉시 반박됨.

**안전한 문장:**
> "같은 frozen pretrained LM 위에서, 같은 extra end-to-end latency budget 아래, LoRA-style parametric write와 verbatim span retention이 서로 경쟁하도록 두고, 어떤 failure mode에서 어느 substrate가 이기며 hybrid가 언제 interior Pareto improvement를 만드는지 실증적으로 밝힌다."

**핵심:** "first hybrid memory" (위험) → "controlled cross-store allocation law under unified budget" (안전). 기존 연구들이 substrate 내 allocation에 머물렀다면, TriStore는 substrate 간 allocation을 다룬다는 구분이 필요하다.

---

## 5. Rebuttal 예상 질문

리뷰에서 거의 반드시 나올 confound 질문들 (리뷰 4, Codex 공통 지적).

| 예상 질문 | 방어 실험 |
|----------|----------|
| "hybrid가 좋은 게 allocation 덕분인가, **프롬프트 토큰이 늘어서**인가?" | Token budget 통제: hybrid와 cache-only가 동일한 materialized prompt length 사용 |
| "write가 좋은 게 allocation 덕분인가, **추가 compute** 덕분인가?" (H5) | Latency-matched comparison: write-only와 cache-only가 동일한 extra latency 사용 |
| "signal pass 자체가 비용인데, overhead 포함하면 **still beneficial**인가?" | Route overhead를 budget에 포함한 net Budget-AUC 보고 |
| "Pareto frontier에서 **seed variance / statistical significance**는?" | Item-level bootstrap CI + paired McNemar + Holm-Bonferroni |
| "왜 external SOTA와 main에서 직접 비교 안 하나?" | Full-context + Naive RAG를 main에 포함. qTTT/GDWM은 최소 appendix |

H5를 방어하려면 동일 budget에서 fixed split (0/100, 50/50, 100/0) vs learned allocation 비교가 필수다.

---

## 6. 필수 실험 목록

### P0: 반드시 (acceptance gating, 이것 없이는 reject)

1. **LoRA write-step 구현 + 작동 입증**: RULER dependency-heavy slice에서 write-only > stable-only
2. **예산 단위 정의 확정**: 1 unit = Δt milliseconds (또는 ΔFLOPs). overhead 포함 공정 비교 프로토콜
3. **Fixed split sweep**: 0/100, 25/75, 50/50, 75/25, 100/0에서 mixed slice accuracy → interior optimum 존재 확인
4. **Full-context baseline**: 모델 context limit 내 bucket에서 main text Pareto figure에 포함
5. **Core ablation 세트:**
   - Allocator 제거: 고정 split vs learned allocation → allocator의 실질 기여 증명
   - Signal pass 제거: random score vs surprisal → signal의 가치 증명
   - K-sweep: preselector sensitivity → method robustness
   - Budget grid별 동일 경향 유지 → 특정 budget에서만 이기는 것이 아님을 증명
6. **다축 회계**: token budget, latency budget, GPU-seconds, route overhead를 같은 표에서 보고

### P1: 강력 권장 (Poster 안정권)

7. **LongBench v2** category별/길이별 Budget-AUC: realistic generalization
8. **혼합 실패 slice 오류 분석**: "왜 hybrid가 필요한가" 정성 + 정량. 어떤 문제에서 write가 실패하고 cache가 구하는지, 반대는 언제인지
9. **Interior split figure**: budget 변화에 따라 optimal write/cache 비율이 이동하는 phase diagram. 논문의 핵심 figure.
10. **Ethics/Privacy:** LoRA에 민감정보 잔류 가능성 실험(canary string test), safety alignment 약화 점검. ICLR 2026은 ethics statement 권장하고 LLM 사용 미공개를 desk reject 사유로 적용.

### P2: 있으면 강해짐 (Spotlight/Oral)

11. **26B A4B partial results**: RULER 핵심 1-2개. "scale-invariant allocation law" 방어
12. **qTTT/GDWM faithful comparison**: appendix라도. qTTT는 ICLR 2026 poster라 부재가 눈에 띔
13. **Allocator dynamics heatmap**: 어떤 chunk가 write에, 어떤 span이 cache에 배분되었는지 시각화. "고유명사/코드(높은 entropy) → cache, 추상적 맥락 → write" 패턴이 보이면 리뷰어 극찬

---

## 7. 논문 집필 지침

### 톤/프레이밍

| 표현 | 판정 | 이유 |
|------|------|------|
| "first hybrid memory" | **위험** | SR-TTT, MemOS, GDWM |
| "controlled cross-store allocation law" | **안전** | 좁고 방어 가능 |
| "novel architecture" | **금지** | frozen LM 위의 method |
| "not RAG" | **금지** | RAG-like 요소 인정하되 contribution 위치 명확히 |
| "consistently outperforms" | **금지** | exact-only slice에서 cache-only가 이길 수 있음 |
| thinking baseline = main enemy | **위험** | qTTT가 이미 "thinking보다 gradient update" 먹었음 |

### 구조 수정 권장

| 현재 계획 | 수정 방향 |
|----------|----------|
| Theory 0.5p | Split Dynamics 분석 1.0p로 전환. theory는 small proposition만 |
| External baseline = appendix only | **main text Pareto figure에 포함** |
| Related Work에 SR-TTT/PERK/GDWM 없음 | **필수 포함** + TTT-E2E, NN-TTT |
| Narrative: "write + cache를 섞었다" | **"allocator가 언제 hybrid advantage를 만드는가"** |
| Thinking baseline 비중 높음 | sub-baseline 수준으로 격하 |
| Ethics statement 없음 | **필수.** privacy, safety alignment, data retention |

### Interior split figure의 위상

리뷰 5: "hybrid가 진짜면 0/100, 25/75, 50/50, 75/25, 100/0 중 어딘가에서 mixed slice에 대한 interior optimum이 보여야 한다. 이게 없으면 H3/H4는 증명되지 않는다."

이 figure 없이는 논문이 성립하지 않는다. X축은 budget (또는 write/cache split ratio), Y축은 accuracy다. Mixed slice에서 0% write도 100% write도 아닌 중간 지점이 최적이라는 것이 핵심 evidence다.

---

## 8. 실행 우선순위

```
Week 1-2: LoRA write 구현 + fail-fast 검증
          ├─ 성공 → fixed split sweep (interior optimum 확인)
          │         └─ interior optimum 보임 → Phase 2
          │         └─ 안 보임 → allocator/signal 재설계
          └─ 실패 → Plan B (query-only, N-step, LoRA-GA)
                    └─ 전부 실패 → pivot (measurement paper)

Week 3-6: RULER sweep + interior split figure + external baselines
          ├─ H3 확인됨 → Phase 3
          └─ core claim 안 보임 → LongBench v2 진행 안 함

Week 7-12: LongBench v2 + Budget-AUC + 26B partial + ethics

Week 8-16: 병렬 집필 (Allocator Dynamics 중심)
```

---

## 9. 리뷰별 활용 가이드

| 소스 | 가장 유용한 용도 | 참조 시점 |
|------|----------------|----------|
| **리뷰 1** | 실행 우선순위 결정, Plan B 옵션 | Week 1-2 계획 수립 |
| **리뷰 2** | 문헌 비교, novelty 경계선, 경쟁 지형 3-tier, allocator 중심 narrative | Introduction/Related Work 집필 |
| **리뷰 4** | P0/P1/P2 분류, confound 질문, ethics/privacy 요건, 베이스라인 표 | 실험 프로토콜 + rebuttal 대비 |
| **리뷰 5** | Claim hierarchy (H3/H4/H5), novelty 좁히기, pivot 판단 기준 | 전체 서사 결정 + Week 6 go/no-go |
| **Codex** | 전체 통합, allocator = intellectual center, 실행 액션 순서 | 종합 로드맵 |

---

## 10. 핵심 한 줄

> **TriStore-BMA의 승부처는 "hybrid memory"라는 말 자체가 아니라, 동일한 예산 아래 write와 cache를 경쟁시켰을 때 mixed-failure slice에서 왜 interior split이 필요한지 실증적으로 증명하는 것이다.**
