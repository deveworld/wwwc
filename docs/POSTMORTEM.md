# TriStore-BMA Postmortem

**결정: 프로젝트 중단 (2026-04-03)**
**사유: Write branch (LoRA inference-time update)가 구조적으로 작동하지 않음**

---

## 1. 무엇을 시도했는가

### 연구 질문
> 고정된 추가 latency 예산 아래에서, long-context 입력의 어떤 정보를 parametric write(LoRA)에, 어떤 정보를 exact cache(verbatim span)에 배분해야 하는가?

### 핵심 가설
Write(LoRA adapter update)와 Cache(exact span retention)를 같은 budget 축에서 경쟁시키면, mixed-failure slice에서 single-store보다 좋은 Pareto frontier를 만든다.

### 전제
Frozen pretrained LM 위에서 inference-time LoRA update로 document 정보를 adapter state에 저장할 수 있다.

---

## 2. 무엇을 발견했는가

### GPU Bring-Up (성공)
- Gemma 4 E2B-it: 로딩, chat template, thinking/non-thinking mode 검증 완료
- Colab T4, A100 MIG 2g-20GB, A100 80GB PCIe에서 테스트
- AutoProcessor + multimodal message format 확정

### Go/No-Go Infrastructure (성공)
- Scaffold quality, cache exact recall, thinking mode: 모두 PASS
- CPU pipeline, CLI, preregistered manifests: 완전 구현

### LoRA Write Branch (실패)

#### 실험 환경
- GPU: NVIDIA A100 80GB PCIe
- Model: google/gemma-4-E2B-it (5.12B params, bf16)
- LoRA: rank=16, alpha=32, targets=language_model q_proj+v_proj (140 params, 3.3M trainable)

#### 버그 수정 과정 (4회 반복)

| 버전 | 버그 | 영향 |
|------|------|------|
| v1 | Document가 prompt에 다 들어감 (ceiling effect) | Stable이 이미 97% → LoRA headroom 없음 |
| v2-a | LoRA가 vision/audio tower에 붙음 | Text NTP gradient가 LoRA에 도달 안 함 |
| v2-b | bf16 precision loss | lr×grad = ~1e-7 → bf16에서 완전 무시 |
| v2-c | Manual SGD 구현 오류 | Weight update 안 됨 |
| v2-d | reset_lora가 lora_A를 zero로 | LoRA output 항상 0 → gradient 0 |
| **v3** | **모든 버그 수정** | **정상 작동 확인** |

#### v3 최종 결과 (모든 버그 수정 후)

| Config | Loss 변화 | Stable | Write | Full | Δ |
|--------|----------|--------|-------|------|---|
| 10-step, lr=1e-3 | 10.10 → 0.12 | 0.000 | 0.000 | 0.875 | +0.000 |
| 20-step, lr=1e-3 | 10.10 → 0.04 | 0.000 | 0.000 | 0.875 | +0.000 |
| 30-step, lr=5e-4 | 10.10 → 0.004 | 0.000 | 0.000 | 0.875 | +0.000 |

- **Loss는 수렴함** (10.10 → 0.004): LoRA가 document의 NTP 패턴을 memorize
- **LoRA weights는 변함** (120/140 nonzero, avg_norm > 0)
- **Generation에서 recall 불가** (write score = 0.000 across all 24 queries × 3 configs)

#### Write 응답 예시

Query: "What is the name of the computational pipeline and its runtime?"
(정답: MolDock-X, 4.2 hours — section 3, scaffold에 없음)

- **Stable:** "Based on the provided text, there is no information available."
- **Write (30-step, loss=0.004):** "I do not have access to the specific details." ← **학습했는데도 recall 못함**
- **Full (doc in prompt):** "MolDock-X, approximately 4.2 hours per protein complex." ← 정상

---

## 3. 왜 실패했는가: Write-Decode Distribution Mismatch

LoRA는 document의 **next-token prediction** 패턴을 학습한다:
```
입력: "Dr. Chen's team developed a novel computational pipeline called"
출력: " MolDock-X"
```

하지만 generation 시 prompt는 완전히 다른 분포다:
```
입력: "[scaffold]\n...\n[query]\nWhat is the computational pipeline?\nAnswer..."
```

LoRA가 학습한 token continuation pattern은 **query-answer 분포에서 활성화되지 않는다.**
이것은 research_plan.md §9.2에서 예측한 정확한 failure mode다.

### 왜 qTTT/PERK/In-Place TTT는 되는가?

이들은 다른 메커니즘을 사용:
- **qTTT:** full context를 KV cache에 유지 + query projection만 update. 정보가 KV cache에 있으므로 distribution mismatch 없음.
- **PERK:** 별도의 retrieval + memory module과 결합. LoRA 단독 recall에 의존하지 않음.
- **In-Place TTT:** architecture 자체를 TTT layer로 교체. Frozen LM이 아님.

즉, **"frozen LM + LoRA write"로는 long-context memory가 되지 않는다.** 작동하는 write 메커니즘들은 더 깊은 아키텍처 변경을 필요로 하며, 그러면 "frozen LM 위의 allocation law"라는 논문의 전제가 무너진다.

---

## 4. 무엇이 유효하게 남는가

### 재사용 가능한 자산
- **CPU pipeline**: chunking, scaffold, preselector, cache, allocator 시뮬레이션 (18 tests passing)
- **TransformersGemmaRunner**: Gemma 4 로딩, chat template, thinking mode (verified)
- **Preregistered manifests**: RULER mixed slices, LongBench v2 selection
- **Budget-AUC framework**: evaluation methodology
- **Gemma 4 API 지식**: AutoProcessor 필수, multimodal message format, thinking tags

### 학술적 교훈
1. **NTP loss 수렴 ≠ 사용 가능한 memory.** LoRA는 token pattern을 memorize하지만, 다른 prompt distribution에서 recall할 수 없다.
2. **Write-decode distribution mismatch는 실제로 치명적이다.** 이론적 경고(§9.2)가 정확히 맞았다.
3. **Fail-fast의 가치.** 4시간 GPU time으로 핵심 가설을 기각할 수 있었다. 수 주간의 실험을 절약.
4. **디버깅은 비용이 크다.** bf16 precision, vision tower LoRA, manual SGD, lora_A zeroing — 4개의 독립적 버그가 결과를 가렸다. 단위 테스트를 먼저 만들어야 한다.

### 관련 연구에 대한 시사점
- **Cache-only allocation** (write 없이 budget 아래 cache 배분)은 여전히 유효한 연구 방향
- **"왜 LoRA write가 long-context recall에서 실패하는가"** 자체가 negative result paper 소재
- **Write-decode mismatch 해결** (scaffold-conditioned training objective 등)이 후속 연구 과제

---

## 5. 타임라인

| 시각 | 작업 |
|------|------|
| 04-03 오전 | 연구 계획 검토, CPU scaffolding 확인 |
| 04-03 오후 | Colab T4 bring-up, A100 MIG go/no-go (PASS) |
| 04-03 저녁 | TransformersGemmaRunner 구현, 외부 리뷰 통합 |
| 04-03 밤 | A100 80GB LoRA write test v1 (ceiling effect 버그) |
| 04-03 밤 | v2 (vision tower + bf16 + SGD 버그) |
| 04-03 밤 | v3 (lora_A zeroing 버그) |
| 04-03 밤 | v3 최종 (모든 버그 수정, loss 수렴, write score 0.000) |
| **04-03 밤** | **FAIL 판정, 프로젝트 중단 결정** |

---

## 6. 수치 기록

### Go/No-Go (A100 MIG 2g-20GB)
- Scaffold quality: PASS (3/3 substrates 정확히 언급)
- Cache exact recall: PASS (hallucination → exact code)
- Thinking mode: PASS (12x latency 차이)

### LoRA Write (A100 80GB PCIe)
- Model: google/gemma-4-E2B-it, 5.12B params
- LoRA: 3.3M trainable (0.06%), rank=16, alpha=32, fp32
- Optimizer: Adam, lr=1e-3 ~ 5e-4
- Training: 10-30 steps per chunk, 8 chunks
- NTP Loss: 10.10 → 0.004 (수렴 확인)
- Write score: 0.000 across 24 queries × 3 configs (recall 실패)
- Full-doc score: 0.875 (정보가 있으면 답할 수 있음 확인)
