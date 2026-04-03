# 외부 리뷰 종합 요약

4개 독립 리뷰(review 1, 2, 4, 5)를 종합한다. review 3은 에이전트 프레임워크 일반론으로 직접 해당하지 않아 제외.

---

## 1. 전체 판정

4개 리뷰 전부 동일한 결론:

> **아이디어/프레이밍은 최상급이나, 현재 상태로는 reject. LoRA write branch 실증이 논문의 생사를 가른다.**

| 리뷰 | 현재 상태 | Write 해결 + baseline 추가 시 | 핵심 표현 |
|------|----------|---------------------------|----------|
| 1 | Borderline~Reject | Spotlight/Oral 가능 | "High Risk, High Return" |
| 2 | 20-35% accept | 상한 이동 | "genuine novelty gap, execution-dependent" |
| 4 | Low (Lean Reject) | Medium (Poster) | "핵심 기여 미구현" |
| 5 | 5-10% (Reject) | 30-45% (Weak Accept~Accept) | "좋은 문제 설정, 증거 없음" |

---

## 2. 전원 합의: 치명적 약점 4개

### 약점 1: Write branch 미검증 (전원 지적, 치명적)

현재 go/no-go check 2는 "더 많은 context를 주면 답이 좋아진다"를 테스트한 것이지, LoRA parametric write를 테스트하지 않았다.

- 리뷰 1: "Write가 안 되면 '복잡하게 꼬아놓은 RAG 논문'으로 전락"
- 리뷰 2: "vanilla LoRA는 full fine-tuning 대비 5-6배 느리게 수렴(LoRA-GA, 2024). 1-step으로는 매우 약한 update만 가능할 수 있다"
- 리뷰 4: "이 상태에서 hybrid와 single-store를 비교하면, 사실상 'cache + prompt engineering vs prompt engineering'에 가까워져 핵심 공헌이 붕괴"
- 리뷰 5: "LoRA 1-step이 안정적으로 안 오르면 methods paper로는 버려야 한다"

**Plan B 옵션 (리뷰어들이 제시):**
- qTTT처럼 Query Projection만 타겟팅 (update surface 축소)
- 1-step이 아닌 N-step gradient update 허용 (budget 단위 재정의)
- LoRA-GA (gradient-aligned initialization) 사용
- Dynamic TTA(2026.02)의 adaptive learning rate
- 실패 시: "budgeted long-context memory study" 또는 negative-result/measurement paper로 pivot

### 약점 2: External baseline 부재 (전원 지적, 심각)

- "Regime이 달라 main에서 비교 안 한다"는 논리는 ICLR에서 통하지 않는다
- ICLR author guide: appendix는 리뷰어가 읽을 의무 없음 → appendix에만 넣는 전략은 위험

**최소 필수 (main text Pareto 그래프에 점으로 표시):**

| 기준점 | 역할 | 비고 |
|--------|------|------|
| Full-context (가능 길이) | "그냥 다 넣으면?" 상한 anchor | 모델 context limit 내 bucket만 |
| Naive RAG (top-K chunk concat) | cache-only 대비 기준선 | token 길이/비용 통제 |

**강력 권장 (main 또는 appendix):**

| 기준점 | 역할 | 비고 |
|--------|------|------|
| qTTT | "pure write" 기준점 | ICLR 2026 poster로 채택됨 — 부재가 눈에 띔 |
| SnapKV 또는 Ada-KV | "pure cache compression" 기준점 | KV-cache 최적화 대표 |
| GDWM | 가장 가까운 경쟁자 | write 내 budget allocation |

리뷰 2의 포지셔닝 제안: "Full-context는 너무 느리고, RAG는 multi-hop에서 무너지지만, Hybrid는 현실적 예산 구간에서 최적 방어선(Pareto frontier)을 형성한다."

### 약점 3: Theory 약함 (4/4 지적, 중간)

toy theory(§4.2)의 expected error 분해 = 기초 볼록최적화. "trivially obvious"라는 공격이 올 것.

**리뷰어들의 대안 제안 (택1):**

- **옵션 A (시각화 전환):** Theory 0.5p 이하로 줄이고, 지면을 "Allocator Split Dynamics 시각화"에 투자. 어떤 종류의 정보(고유명사/코드 = 높은 entropy vs 추상적 맥락)를 만났을 때 allocator가 cache를 택하고 write를 택하는지 heatmap으로 보여준다. (리뷰 1)
- **옵션 B (최소 이론 + 측정학):** allocation을 multi-objective knapsack으로 정식화하고, 단조성/상계 같은 성질 1-2개만 증명. Budget-AUC를 "평가 기여"로 격상. (리뷰 4)
- **핵심 조건:** theory가 "hybrid가 dominate한다"만 말하면 안 되고, **"어떤 failure-mode mixing 조건에서 hybrid가 dominate하는가"**를 characterize해야 함. (리뷰 2)

### 약점 4: 모델 규모 (3/4 지적, 보통)

E4B(5B) 단독은 generalization 의심을 받을 수 있다.

**최소 방어:** RULER 핵심 1-2개를 26B A4B로 돌려 appendix에 "Scale-invariant allocation law" 한 문단 작성.

**완화 요인:** qTTT도 Qwen3-4B에서 했고, PERK도 0.5B/7B, ICLR 2026에서 5B급 실험이 fatal하지는 않다. (리뷰 2, 5)

---

## 3. research_plan.md에 없는 경쟁 위협

리뷰 2와 5가 발견한, 현재 계획서에 반영되지 않은 관련 연구들:

### 직접적 경쟁자 (Related Work 필수 포함)

| 논문 | 공개일 | 위협 수준 | 이유 |
|------|--------|---------|------|
| **SR-TTT** | 2026.02 | **높음** | "highly surprising tokens만 residual cache에 보내는 hybrid memory" — 가장 직접적 경쟁자. cache의 필요성을 write 쪽에서 인정한 논문 |
| **PERK** | ICLR 2026 poster | **높음** | "LoRA를 test-time memory module로 사용" — write substrate의 기존 연구. 0.5B/7B에서 강한 결과 |
| **GDWM** | 2026.01 | **높음** | budget-constrained memory consolidation. write 내 allocation을 다루지만 cross-substrate는 안 함. TriStore가 GDWM을 subsume |
| **TTT-E2E** | 2025.12 | 중간 | LoRA-based continual learning, constant latency. exact recall 실패 인정 — cache 필요 근거 |
| **MemOS** | 2025.07 | 중간 | plaintext/activation/parameter 3-tier memory with distillation — 개념적 중복 |
| **Latent Context Compilation** | 2026.01 | 중간 | disposable LoRA로 long context compile |
| **GradMem** | 2026.03 | 중간 | memory tokens을 gradient descent로 최적화 |

### Novelty 방어선 수정 필요

리뷰 5의 경고: **"write vs cache를 섞는다"는 큰 문장 자체가 더 이상 안전한 novelty claim이 아니다.**

현재 research_plan.md의 novelty sentence:
> "기존 문헌이 write/update, exact retention, architecture-level memory, KV lifecycle evaluation을 각각 따로 발전시켰다면, 본 논문은 frozen LM 위에서 selective write와 selective exact cache를 같은 extra-latency budget 안에서 함께 회계하고..."

**수정 권장:**
> "같은 frozen pretrained LM 위에서, 같은 extra end-to-end latency budget 아래, LoRA-style parametric write와 verbatim span retention이 서로 경쟁하도록 두고, 어떤 failure mode에서 어느 substrate가 이기며 hybrid가 언제 interior Pareto improvement를 만드는지 실증적으로 밝힌다."

핵심 변화: "아무도 hybrid memory를 안 했다" (위험) → "controlled cross-store allocation law under unified budget" (안전)

---

## 4. 리뷰에서 추출한 필수 실험 목록

### P0: 반드시 (acceptance gating)

1. **LoRA write-step 구현 + 작동 입증** — RULER dependency-heavy slice에서 write-only > stable-only
2. **예산 단위 정의 확정** — 1 unit = Δt milliseconds (또는 ΔFLOPs). Overhead 포함 공정 비교
3. **Full-context baseline** — 모델 context limit 내 bucket에서 main text에 포함
4. **Core ablation 세트:**
   - Allocator 제거: 고정 split (100/0, 0/100, 50/50) vs learned allocation
   - Signal pass 제거: random score vs surprisal
   - K-sweep: preselector sensitivity
   - Budget grid별 동일 경향 유지 확인

### P1: 강력 권장 (Poster 안정권)

5. **LongBench v2** category별/길이별 Budget-AUC
6. **혼합 실패 slice 오류 분석** — "왜 hybrid가 필요한가" 정성 + 정량
7. **Interior split figure** — budget 변화에 따라 optimal write/cache 비율이 이동하는 phase diagram
8. **Ethics/Privacy:** LoRA에 민감정보 잔류 가능성 실험 (canary string test), safety alignment 약화 점검

### P2: 있으면 강해짐 (Spotlight/Oral)

9. **26B A4B partial results** — "scale-invariant allocation law" 방어
10. **qTTT/GDWM faithful comparison** — appendix라도
11. **Allocator dynamics heatmap** — 어떤 chunk가 write에, 어떤 span이 cache에 배분되었는지 시각화

---

## 5. 논문 집필 시 주의사항

### 톤/프레이밍

- "first hybrid memory" → **위험**. SR-TTT, MemOS, GDWM이 있음
- "controlled cross-store allocation law under unified budget" → **안전**
- "novel architecture" → **금지**
- "not RAG" → **금지** (RAG-like 요소를 인정하되 contribution 위치를 명확히)
- thinking baseline을 "main enemy"로 두지 않기 — qTTT가 이미 "thinking보다 gradient update가 낫다"를 먹었음

### 구조 수정 권장

- Theory 0.5p → **Split Dynamics 분석 1.0p로 전환**
- External baseline을 appendix가 아닌 **main text Pareto figure에 포함**
- Ethics statement 필수 (ICLR 2026은 LLM 사용 미공개 + 환각 reference를 desk reject 사유로 적용)
- Related Work에 SR-TTT, PERK, GDWM, TTT-E2E 반드시 포함

### Interior split figure = 논문의 심장

리뷰 5: "hybrid가 진짜면 0/100, 25/75, 50/50, 75/25, 100/0 중 어딘가에서 mixed slice에 대한 interior optimum이 보여야 한다. 이게 없으면 H3/H4는 증명되지 않는다."

이 figure가 없으면 **논문이 성립하지 않는다.** Budget 축에서 write/cache split이 이동하는 모습이 핵심 evidence.

---

## 6. 실행 우선순위 최종 정리

```
Week 1-2: LoRA write 구현 + fail-fast 검증
          ├─ 성공 → Phase 2 진행
          └─ 실패 → Plan B (query-only update, N-step) 시도
                    └─ 전부 실패 → pivot (measurement paper)

Week 3-6: RULER sweep + interior split figure + external baselines
          └─ core claim 안 보이면 → LongBench v2 진행 안 함

Week 7-12: LongBench v2 + Budget-AUC + 26B partial

Week 8-16: 병렬 집필 (Split Dynamics 중심)
```

**가장 중요한 한 문장 (리뷰 5):**

> "이 연구는 가치 있다. 하지만 현재는 '좋은 문제 설정' 단계이고, ICLR급 accept를 받으려면 'hybrid interior split이 왜 실재하며 왜 중요한가'를 실제 write update로 증명하는 순간까지 가야 한다."
