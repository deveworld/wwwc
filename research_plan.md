# TriStore-BMA Final Research Plan (v5.1 / Gemma 4 lock)

작성일: 2026-04-03  
상태: **최종 잠금본 / 구현 시작용**  
제출 목표: **ICLR main track**  
핵심 정체성: **budgeted test-time memory allocation paper**

---

## 0. Executive summary

이 프로젝트의 최종 목표는 “새로운 거대 장문맥 아키텍처”를 제안하는 것이 아니다. 이 논문의 핵심 질문은 다음 하나로 고정한다.

> **고정된 추가 추론 예산 아래에서, long-context 입력의 어떤 정보는 parametric write에, 어떤 정보는 exact cache에 배분해야 하는가? 그리고 hybrid allocation이 언제 single-store strategy보다 더 좋은 accuracy–efficiency Pareto frontier를 만드는가?**

최종 메인 클레임도 이 질문에 맞춰 절제해서 쓴다.

> **동일한 frozen language model, 동일한 stable scaffold, 동일한 프롬프트/디코딩 조건, 그리고 동일한 추가 end-to-end latency 예산 아래에서, selective write와 selective exact cache를 함께 쓰는 hybrid allocator는 mixed-failure long-context slice에서 write-only 및 cache-only specialization보다 더 좋은 Pareto frontier와 더 높은 Budget-AUC를 형성한다.**

이 계획은 다음 사실 위에서 구성된다.

1. ICLR 2026은 **초기 제출 본문 9페이지 제한**, **부록은 리뷰어가 읽을 의무 없음**, **익명성 위반 시 desk reject**, **significant LLM usage 공개 의무**, **저자 리뷰 등록 누락 시 desk reject 가능성**을 명시한다. 따라서 본문은 한 문장짜리 과학적 질문과 그 질문을 닫는 증거 중심으로 구성한다.
2. ICLR reviewer guide는 accept 판단의 핵심을 **문제의 명확성, 문헌 속 위치, claim support의 rigor, 그리고 community에 주는 새로운 지식**으로 둔다. **SOTA가 아니어도** 새롭고 영향력 있는 지식을 설득력 있게 보여주면 가치가 있다고 명시한다.
3. long-context 문헌은 이미 qTTT, GDWM, SR-TTT, In-Place TTT, Titans, ATLAS, GradMem, WG-KV, SCBench, HELMET 등으로 빠르게 차고 있다. 따라서 이 논문이 살아남으려면 “memory를 여러 개 붙였다”가 아니라, **무엇을 언제 write하고 언제 exact cache해야 하는지에 대한 allocation law**를 보여줘야 한다.

이 최종본은 지난 모든 리뷰를 반영해, 구조적 수정이 아니라 **구현 전에 잠가야 할 사양, subtle risk, paper tone, go/no-go, rebuttal 준비**까지 포함한다.

추가로, backbone family는 이제 **Gemma 4 instruction-tuned open models**로 잠근다.  
메인 bring-up은 **Gemma 4 E2B / E4B**, 메인 결과 후보는 **Gemma 4 E4B**, 확장 후보는 **Gemma 4 26B A4B**, 31B dense는 재현성/비용 이유로 메인 기본 대상에서 제외한다.

---

## 1. 최종 논문 포지셔닝

### 1.1 이 논문이 아닌 것

이 논문은 다음으로 읽히면 안 된다.

- 새로운 recurrent / hybrid backbone 논문
- large-scale from-scratch pretraining 논문
- pure RAG 논문
- pure KV-compression 논문
- pure TTT 개선 논문
- 모든 최신 long-context baseline을 정면으로 이기는 SOTA 논문

### 1.2 이 논문인 것

이 논문은 다음으로 읽혀야 한다.

- **frozen pretrained LM 위의 test-time allocation law** 논문
- **write vs exact cache**의 trade-off를 같은 예산 축으로 회계하는 논문
- **mixed-failure long-context tasks**에서 hybrid allocation이 필요한 이유를 empirical + theoretical로 보이는 논문
- **bounded materialized prompt**를 사용하지만, contribution은 retrieval 자체가 아니라 **allocation principle**에 있다는 논문

### 1.3 RAG-like 구조에 대한 최종 stance

이 프로젝트는 bounded final prompt를 만들기 위해 stable scaffold와 selected cache spans를 materialize한다. 따라서 구조적으로는 **RAG-like**하다는 공격이 들어올 수 있다. 이 점은 부정하지 않는다. 오히려 본문에서 먼저 인정한다.

다만 차이는 분명히 적는다.

- 기존 RAG의 핵심 공헌은 **retrieval-augmented generation 자체**다.
- 이 논문의 핵심 공헌은 **고정된 추가 latency 예산 아래, exact retention(materialized evidence)과 parametric adaptation(write) 사이의 allocation law**다.
- retrieval/scaffold는 **stable always-on substrate**의 구현 수단일 뿐이며, 새로운 claim의 중심은 아니다.

이 framing의 목적은 novelty의 위치를 명시적으로 지정하는 데 있다.

---

## 2. 관련 문헌 속 최종 위치

### 2.1 write/update 중심 축

**qTTT**는 long-context에서 thinking-token 기반 inference-time scaling이 빠르게 한계에 도달한다고 보고하고, query-only test-time training이 더 효율적이라고 주장한다. 핵심은 **context-specific parametric adaptation**이 중요하다는 점이다. 하지만 qTTT는 기본적으로 write/update 쪽에 서 있고, exact token retention 자체를 주제로 삼지 않는다.

**GDWM**은 test-time adaptation을 budget-constrained memory consolidation 문제로 재정식화한다. 즉, 어디에 몇 step의 write를 배분할지를 중심으로 본다. 이 논문이 가장 직접적으로 계승하는 문헌은 GDWM이다. 단, GDWM은 write 쪽 budget allocation이 중심이고, **cache/write 간 cross-substrate allocation**은 다루지 않는다.

**In-Place TTT / LaCT**는 TTT의 hardware inefficiency를 줄이고, chunk-wise update가 practical하다는 점을 보여준다. 이 논문은 이 축에서 “write를 하되, 반드시 chunk-wise and hardware-aware하게 해야 한다”는 engineering lesson을 가져온다.

### 2.2 exact retention / cache 중심 축

**SR-TTT**는 compressed fast-weight memory가 exact recall에서 catastrophic failure를 일으킬 수 있고, surprisal-driven residual cache가 이를 완화할 수 있다고 보여준다. 이 논문이 exact cache branch를 두는 가장 직접적인 근거다.

**Write-Gated KV (WG-KV)**는 long-context inference를 KV admission / selection / eviction이라는 causal memory management로 본다. 이 논문은 exact cache를 trainable gate로 두는 KV-centric 접근이지만, 우리와 달리 **parametric write branch**를 함께 budget allocation 문제로 보지는 않는다.

### 2.3 architecture-level memory 축

**Titans**와 **ATLAS**는 memory-augmented architecture family다. 둘 다 test-time memory를 배우거나 장기 memory module을 architecture에 내장한다. 이들은 “hybrid memory가 필요할 수 있다”는 큰 방향성에는 가깝지만, 우리의 contribution과는 다르다. 우리는 architecture 자체를 바꾸지 않고, **pretrained frozen LM 위에서 inference-time allocation policy**를 다룬다.

**GradMem**은 memory tokens에 context를 gradient descent로 써 넣는 방식이다. 이는 write의 또 다른 형태를 보여주며, “무언가를 parametric or pseudo-parametric state에 써 넣는 것”이 long context를 압축하는 유효한 수단이라는 근거가 된다. 하지만 역시 exact cache branch와의 allocation law는 직접 다루지 않는다.

### 2.4 evaluation-design 축

**RULER**는 retrieval-only가 아니라 multi-hop tracing과 aggregation까지 포함하는 synthetic long-context benchmark다. 이 논문에서 failure mode decomposition을 가장 깨끗하게 해준다.

**LongBench v2**는 현실형 long-context multitask benchmark로, 503개 MCQ와 8k–2M words, 6개 task category를 제공한다. 이 논문은 real-world mixed failure를 보는 메인 현실형 벤치로 쓰기에 적합하다.

**ZeroSCROLLS**는 zero-shot long-text understanding benchmark다. 다만 이번 논문의 main claim은 “mixed-failure allocation law”이므로, ZeroSCROLLS는 메인보다는 support/appendix 역할이 더 적절하다.

**HELMET**은 synthetic NIAH만으로는 downstream capability를 제대로 평가하기 어렵다고 지적한다. 이 논문은 우리가 왜 RULER 하나로 끝내지 않고 LongBench v2를 함께 쓰는지의 justification이 된다.

**SCBench**는 shared-context multi-request / KV lifecycle 문제를 다룬다. 유용하지만 이번 논문의 main regime는 **single-query, single-request**이므로 메인 claim 바깥으로 둔다.

### 2.5 최종 novelty sentence

최종적으로 이 논문의 새로움은 다음 한 문장으로 정리한다.

> **기존 문헌이 write/update, exact retention, architecture-level memory, KV lifecycle evaluation을 각각 따로 발전시켰다면, 본 논문은 frozen LM 위에서 selective write와 selective exact cache를 같은 extra-latency budget 안에서 함께 회계하고, mixed-failure long-context regime에서 왜 그 split이 중요한지를 theory와 equal-budget Pareto로 보여준다.**

---

## 3. 최종 문제정의

입력 문서 전체를 raw document라 하고, 질의/질문을 \(q\)라 하자. raw document는 길이가 매우 길 수 있으나, 최종적으로 LM에 materialize되는 prompt는 bounded하다.

### 3.1 세 substrate

1. **Stable scaffold** \(S\)  
   - always-on
   - raw document 전체에서 cheap retrieval/extraction으로 만든 bounded support context
   - 모든 baseline과 method가 동일하게 공유

2. **Write state** \(W\)  
   - frozen base model 위의 small fast-weight adapter state (예: LoRA/adapters)
   - chunk-level self-supervised loss로 inference-time update

3. **Exact cache** \(C\)  
   - raw document에서 골라낸 exact spans/tokens
   - 최종 prompt에 명시적으로 materialize

### 3.2 allocation 변수

raw document를 \(m\)개 chunk \(C_1,\dots,C_m\)로 나눈다.

- write allocation: 각 chunk에 대해 write step 수 \(k_i \in \mathbb{Z}_{\ge 0}\)
- cache allocation: 각 후보 span \(s\)에 대해 binary keep variable \(m_s \in \{0,1\}\)

### 3.3 예산 제약

이 논문의 **메인 budget axis는 extra end-to-end latency**다.  
즉, stable-only pipeline의 latency를 \(\tau_{stable}\), 어떤 방법의 latency를 \(\tau_m\)라 할 때

\[
B = \tau_m - \tau_{stable}
\]

로 정의한다.

secondary logging은 다음을 별도로 기록한다.

- extra GPU-seconds
- total write steps
- total cached tokens/spans
- route/signal overhead
- decode overhead

부록에서는 compute vs cache-size 2D Pareto도 제공한다.

### 3.4 왜 raw document length와 materialized prompt length를 분리하는가

이 논문에서 internal method는 raw document 전체를 chunk-wise로 읽을 수 있지만, 최종적으로 LM에 들어가는 prompt는 bounded scaffold + cached spans + query로 제한된다.  
즉,

- **raw document length**: allocator가 볼 수 있는 전체 원문 길이
- **materialized prompt length**: 실제 base LM이 마지막 decode 시 보는 prompt 길이

를 구분한다.

이 구분이 중요한 이유는 다음과 같다.

- internal hybrid/write/cache methods는 bounded materialized prompt regime에서 동작한다.
- 반면 full-context baselines(thinking, direct in-context, qTTT appendix version)는 실제 prompt 길이가 모델 context limit에 직접 묶인다.

이 차이를 문서 전체에서 명시해야 E2B/E4B/26B A4B와 32k/64k/128k의 혼선이 사라진다.

### 3.5 backbone family lock: Gemma 4

이 계획서의 기본 backbone family는 **Gemma 4 instruction-tuned models**로 고정한다.

- **main calibration ladder:** E2B -> E4B
- **main paper default:** E4B
- **appendix / stretch candidate:** 26B A4B
- **not default:** 31B dense

이 선택의 이유는 다음과 같다.

- Gemma 4는 공식적으로 **configurable thinking modes**, **native system role**, **latest Transformers support**, **LoRA/QLoRA tuning path**, **128K/256K context window**를 제공한다.
- E2B/E4B는 bring-up과 debugging이 쉽고, write branch calibration을 빠르게 닫기에 적합하다.
- E4B는 메인 연구 질문을 single-GPU에 가깝게 재현하면서도 E2B보다 reasoning headroom이 크다.
- 26B A4B는 더 강한 성능 후보지만, 메인 claim을 닫기 전에 올리면 route/write latency accounting과 adapter engineering 리스크가 커진다.

운영 원칙:

- 한 실험 run 안에서는 **오직 하나의 Gemma 4 variant만 사용**한다.
- stable-only / thinking / write-only / cache-only / hybrid는 **항상 동일 variant**를 공유한다.
- 메인 텍스트의 주 결과는 **E4B 기준**으로 쓴다.
- E2B는 bring-up / robustness / low-cost 재현성 evidence로 사용한다.
- 26B A4B는 core claim이 닫힌 뒤 appendix 또는 support table로 확장한다.

---

## 4. 최종 가설과 toy theory

### 4.1 정성 가설

- **H1:** pure exact-recall slices에서는 cache-only가 write-only보다 강하다.
- **H2:** pure diffuse dependency slices에서는 write-only가 cache-only보다 강하다.
- **H3:** mixed-failure slices에서는 hybrid가 single-store보다 더 좋은 Pareto frontier를 만든다.
- **H4:** 최적 split은 task mixture에 따라 이동한다.
- **H5:** hybrid의 이득은 “더 많은 compute를 썼기 때문”이 아니라, **예산을 다른 substrate에 나누어 쓴 결과**다.

### 4.2 toy theory

expected error를 다음처럼 분해한다.

\[
\mathbb{E}[\mathrm{err}(b_w,b_c)] = p_B\,\varepsilon_B(b_w) + p_C\,\varepsilon_C(b_c) + const,
\]

여기서

- \(b_w\): write에 쓴 예산
- \(b_c\): cache에 쓴 예산
- \(p_B, p_C\): failure mode mixture weights
- \(\varepsilon_B, \varepsilon_C\): 각 substrate의 diminishing-return error curve

를 뜻한다.

\(\varepsilon_B, \varepsilon_C\)가 단조감소·오목(concave)이고, single-store가 제거하지 못하는 구조적 residual error가 존재하면, mixed regime에서 interior optimum이 존재하고 hybrid split이 extremal allocation(write-only or cache-only)을 지배하는 budget 구간이 생긴다.

### 4.3 theory가 만들어야 하는 검증 가능한 예측

- mixed ratio가 바뀌면 optimal split이 이동해야 한다.
- exact-heavy slice에서는 cache marginal utility가 더 커야 한다.
- dependency-heavy slice에서는 write marginal utility가 더 커야 한다.
- synthetic oracle allocation과 heuristic allocation 사이 gap은 줄어들어야 한다.

메인 논문에서는 theorem을 크게 밀지 말고, **small but falsifiable proposition**으로 유지한다.

---

## 5. 최종 method: TriStore-BMA

## 5.1 high-level pipeline

1. raw document를 chunk로 나눈다.
2. cheap preselector가 전체 문서에서 상위 \(K\)개 candidate chunk를 고른다.
3. shortlisted chunks에 대해서만 exact signal pass를 돌린다.
4. marginal-utility interleaving allocator가 다음 한 단위 예산을 write에 쓸지 cache에 쓸지 고른다.
5. 최종적으로 stable scaffold + cached spans + query를 materialize하고, write state를 적용한 frozen LM으로 답을 decode한다.

### 5.2 stable scaffold

stable scaffold는 이 논문의 **항상 켜진 기본 substrate**다.

구성 원칙:

- query lexical overlap
- cheap embedding similarity
- document coverage rule
- fixed number of scaffold spans or fixed total scaffold token budget

중요한 fairness rule:

- stable scaffold policy는 **Week 2 이후 전 실험에서 완전 고정**한다.
- benchmark별 retuning 금지
- stable-only / write-only / cache-only / hybrid가 **동일 scaffold**를 공유

### 5.3 signal pass: 최종 수정본

최신 리뷰에서 가장 중요했던 이슈는 chunk-local forward artifact였다.  
chunk를 독립적으로 넣고 loss를 계산하면, chunk 시작 부분이 context 부재 때문에 인위적으로 어렵게 보일 수 있다.

따라서 final plan에서는 다음을 기본으로 고정한다.

#### 기본 signal pass

각 shortlisted chunk \(C_i\)에 대해

\[
[\text{scaffold prefix}] + [C_i]
\]

를 모델에 넣고, **loss/surprisal는 오직 \(C_i\) 부분에서만** 계산한다.

이렇게 하면 signal의 해석이

> “stable scaffold로도 충분히 커버되지 않는 residual difficulty”

가 되어 story와 일치한다.

#### optional sensitivity

부록 sensitivity:

- sliding overlap prefix
- chunk-start token masking (예: 앞 64 tokens 제외)

하지만 메인 default는 scaffold-prefixed chunk loss로 고정한다.

### 5.4 write branch

write는 chunk-level self-supervised NTP/reconstruction loss를 줄이는 방향으로 small adapter state를 update한다.

#### 기본 구현

- frozen base LM
- small LoRA/adapters on selected layers
- 1-step or few-step gradient update per selected chunk
- chunk-wise update only

#### 왜 LoRA/adapters인가

- frozen pretrained LM 유지 가능
- update surface 작음
- single GPU 재현 가능
- In-Place TTT/LaCT 계열의 “practical test-time update” 철학과 정렬
- Gemma 4는 공식 문서 차원에서 LoRA 및 QLoRA tuning path를 제공한다.

#### 반드시 확인해야 할 subtle risk

write는 raw chunk에 대해 학습하지만, final decode는 scaffold + cache + query prompt distribution에서 일어난다.  
즉, **write-update distribution과 final decode distribution이 다를 수 있다.**

이 문제는 Tier 1/Week 1–2에서 곧바로 검증한다.

- dependency-heavy synthetic slice에서 write-only가 stable-only를 실제로 이기는가?
- 만약 전혀 못 이기면, 현재 write objective가 “압축 가능한 지식 저장”으로 작동하지 않는다는 뜻이다.

이 경우 fallback은 두 가지다.

1. write objective를 scaffold-conditioned objective로 더 바꾼다.
2. write branch를 메인 method의 중심이 아니라 auxiliary로 낮춘다.

write는 계획상 필수 구성요소지만, 효과가 확인되지 않으면 스토리에서 중심을 내린다.

### 5.5 cache branch

cache는 exact span을 final prompt에 보존하는 branch다.

기본 원칙:

- signal은 surprisal 중심
- exact span length는 bounded
- top spans are materialized verbatim
- redundancy penalty는 메인 method에서 제거하고, appendix ablation으로만 둔다

메인 default는 단순하게 유지한다.

\[
U_c(s) = \tilde S(s)
\]

where \(\tilde S(s)\) is normalized span surprisal score.

### 5.6 utility simplification

최종본에서는 hyperparameter 수를 줄인다.

메인 default:

\[
U_w(i,r)=\tilde L_i - \delta (r-1)
\]

\[
U_c(s)=\tilde S(s)
\]

- \(\tilde L_i\): scaffold-conditioned chunk loss
- \(r\): same chunk에 누적 write step index
- \(\delta\): repeated write diminishing-return penalty (single scalar)

query relevance, coverage, redundancy는 메인 utility에 다 때려 넣지 않고,

- preselector 단계
- hard constraints
- appendix ablation

으로 분리한다.

### 5.7 allocator

write-first greedy를 버리고, **marginal-utility interleaving**을 메인 allocator로 고정한다.

각 step에서

- next write step의 marginal gain
- next cache span의 marginal gain

을 비교하고 더 큰 쪽에 한 단위 예산을 배분한다.

#### 이유

- write-first는 write 편향을 내장한다.
- theory의 interior optimum / KKT 해석과 안 맞는다.
- interleaving은 “split matters”를 method에서 구현하는 직관적인 형태다.

### 5.8 최종 prompt

최종 decode prompt는 아래와 같이 bounded하게 구성한다.

\[
\text{Prompt} = [\text{instruction}] + [q] + [S] + [C]
\]

write state는 prompt에 materialize되지 않고 adapter state로만 반영된다.

---

## 6. 최종 baseline 전략

### 6.1 메인 baselines (internal only)

1. **Stable-only**  
   동일 scaffold, no write, no cache

2. **Thinking baseline**  
   Gemma 4 native thinking mode 사용, extra latency matched by `max_new_tokens`

3. **Write-only**  
   same scaffold, write on, cache off

4. **Cache-only**  
   same scaffold, write off, cache on

5. **Hybrid**  
   same scaffold, both on

6. **Random allocation**  
   same budget, random write/cache choice

7. **Oracle (synthetic only)**  
   synthetic mixed tasks에서 정답 정보에 기반한 upper bound split

### 6.2 왜 external SOTA head-to-head를 메인에서 뺄까

- qTTT는 full-context regime
- GDWM은 write policy 중심
- SR-TTT는 TTT+residual cache 구조
- WG-KV는 KV admission 문제
- Titans/ATLAS는 architecture-level memory
- SCBench는 shared-context KV lifecycle

즉, 비교 자체가 무의미한 것은 아니지만, **같은 문제/같은 substrate/같은 regime/같은 cost model**이 아니다. faithful reproduction 없이는 main table에서 직접 붙이는 것이 오히려 약점이 된다.

따라서 external baselines는 다음으로 제한한다.

- **main text:** mention only
- **appendix:** strongest one or two if faithful reproduction succeeds

### 6.3 thinking baseline의 최종 정의

Gemma 4 공식 model card 기준으로 전 모델이 **configurable thinking modes**를 지원한다. Transformers example에서는 `processor.apply_chat_template(..., enable_thinking=True/False)`로 제어하고, `processor.parse_response()`로 thought/final answer를 분리한다. 또한 26B A4B와 31B는 thinking off에서도 empty thought block tag를 낼 수 있으므로, thinking baseline 비교에서는 **visible output parsing과 latency accounting을 함께 고정**해야 한다.

따라서 최종 정의는 다음과 같다.

- **main text thinking baseline:** Gemma 4 native thinking mode + latency-matched `max_new_tokens`
- **main text non-thinking baselines:** same chat template with `enable_thinking=False`
- **appendix sensitivity:** E4B vs E2B thinking gain, and if time permits 26B A4B thinking sensitivity
- **internal baselines:** thinking baseline을 제외하고 모두 non-thinking mode

이렇게 하면 “thinking baseline이 부정확하다”는 공격을 피할 수 있다.

---

## 7. 벤치마크와 slice 설계

## 7.1 메인 벤치: RULER

RULER는 retrieval, multi-hop tracing, aggregation, QA를 포함하는 synthetic benchmark로, failure mode를 분해하기 가장 좋다.

### 메인 task selection

최종 고정 후보:

- retrieval: `niah`
- multi-hop: `variable_tracking`
- aggregation: `common_words_extraction` or `freq_words_extraction`
- QA: one QA task

### raw length buckets

- 32k
- 64k
- 128k

중요: 여기서 length는 **raw document length**다.  
internal methods의 materialized prompt는 bounded하므로, Gemma 4 E2B/E4B(128K context)에서도 raw 64k/128k를 다룰 수 있다.

### mixed-failure slice preregistration

Week 2에 아래 mixed slices를 고정하고, 이후 실험 결과와 무관하게 그대로 쓴다.

- retrieval + tracing
- retrieval + aggregation
- tracing + QA

### 왜 preregistration이 필요한가

실험 후에 mixed slice를 다시 정의하면 cherry-picking 공격을 받기 쉽다.  
따라서 mixed slice는 **성공/실패와 무관하게 미리 잠근다.**

## 7.2 메인 현실형 벤치: LongBench v2

LongBench v2는 503 MCQ, 8k–2M words, 6 category를 가진 현실형 장문맥 벤치다.

### 메인 사용 범위

메인 논문에서는 다음 범주를 우선한다.

- multi-document QA
- code repository understanding
- long structured data
- long dialogue history or single-document QA 중 하나

### why these slices

이들은 exact token retention과 diffuse dependency reasoning이 동시에 필요한 mixed regime를 만들기 쉽다.

### filtering policy

- raw document length bucket을 먼저 기록
- internal methods는 bounded prompt regime로 평가
- appendix의 full-context baselines는 model context fit subset만 사용

즉, **internal comparison은 raw-length 기준**, **external full-context comparison은 fit-subset 기준**으로 분리한다.

## 7.3 ZeroSCROLLS

ZeroSCROLLS는 zero-shot long-text benchmark이며, train set이 없고 test와 small validation 중심이다.

이번 논문에서의 역할:

- 메인 claim 검증용이 아니라
- **supporting generalization evidence**

후보 task:

- GovReport
- Qasper
- NarrativeQA (preferable over weaker long-context slices)

단, 일정이 빡빡하면 **appendix only**로 둔다.

## 7.4 HELMET과 SCBench의 역할

- **HELMET:** benchmark philosophy reference  
  (synthetic-only로 충분하지 않다는 근거)
- **SCBench:** excluded-from-main regime  
  (shared-context / KV lifecycle / multi-request는 이번 main claim 바깥)

---

## 8. 비용 회계와 metric

### 8.1 메인 budget axis

메인 budget axis는 **extra end-to-end latency**로 고정한다.

\[
B = \tau_{method} - \tau_{stable}
\]

같은 hardware, 같은 software stack, batch size 1, 동일 decode setting에서 측정한다.

### 8.2 provisional budget grid

초기 후보는

\[
B \in \{0,
0.25,
0.5,
1.0\}\times \tau_{stable}
\]

이다.

하지만 최신 리뷰가 지적했듯, signal pass 비용이 작지 않으면 작은 B에서 routing만 하고 write/cache를 거의 못 쓰는 상황이 생길 수 있다. 따라서 **Week 1–2 calibration 후 grid를 최종 잠근다.**

### 8.3 budget calibration decision rule

- default K에서 median route cost가 **medium budget의 40%**를 넘으면:
  - K를 줄이거나
  - low budget point를 버리고 `{0, 0.5, 1.0, 1.5} × τ_stable`로 이동
- route cost가 **small budget의 80%**를 넘으면 그 small-B 점은 main figure에서 제거

즉, budget grid도 data-driven하게 early calibration 후 lock한다.

### 8.4 main metric

메인 metric은 **Budget-AUC**로 둔다.

이유:

- 단일 budget point 승패는 불안정하다.
- Pareto frontier 전체를 한 숫자로 요약할 수 있다.
- “작은 B에서는 불리하지만 중간 B에서 크게 이긴다” 같은 구조를 반영할 수 있다.

보조 metric:

- low / medium / high budget point score
- category score
- mixed-slice score
- oracle gap closure

### 8.5 statistics

- item-level bootstrap confidence interval
- paired McNemar test (hybrid vs write-only, hybrid vs cache-only)
- multiple comparisons: Holm–Bonferroni

LongBench v2 같은 MCQ evaluation에서는 이 구성이 가장 방어적이다.

---

## 9. subtle risks와 최종 go/no-go

이 섹션은 이번 최종본의 가장 중요한 차별점이다. 구조적 문제는 이미 끝났고, 이제는 **구현 단계에서 터질 수 있는 subtle한 위험을 미리 문서화**해야 한다.

### 9.1 Risk A: chunk-local forward artifact

**문제:** chunk를 단독으로 loss 평가하면 시작 부분 loss가 인위적으로 높아진다.

**고정된 해결책:** scaffold-prefixed chunk loss.

**Go/No-Go:** 
- scaffold-prefixed signal과 chunk-only signal이 크게 다르면, chunk-only는 폐기.
- main text는 scaffold-prefixed만 유지.

### 9.2 Risk B: write–decode distribution mismatch

**문제:** write는 raw chunk distribution에서 학습하지만, final decode는 scaffold+cache+query distribution에서 일어난다.

**Week 1–2 validation:**
- dependency-heavy synthetic slice에서 write-only가 stable-only를 이기는지 먼저 확인

**Go/No-Go:**
- medium budget에서 write-only가 dependency-heavy slice에서 stable-only 대비 일관된 이득을 전혀 못 내면:
  1. write objective 수정 또는
  2. write branch의 역할 축소

### 9.3 Risk C: route overhead dominates small budgets

**문제:** shortlist exact signal pass가 budget을 먹어버릴 수 있다.

**해결:** cheap preselector + top-K exact pass + budget recalibration.

**Go/No-Go:**
- median route overhead > small budget의 80% 이면 small-B 제거
- median route overhead > medium budget의 40% 이면 K down 또는 budget grid 확대

### 9.4 Risk D: K sensitivity가 너무 크다

**문제:** shortlist size K가 결과를 크게 바꾸면 method가 fragile해 보인다.

**정책:** K sensitivity는 appendix-only가 아니라 **main analysis inset**으로 승격.

**Go/No-Go:**
- default K 주변에서 ranking이 완전히 뒤집히면, current K choice is unstable → preselector 재설계

### 9.5 Risk E: scaffold가 너무 약하거나 너무 강하다

**문제:** scaffold가 너무 약하면 trivial gain, 너무 강하면 hybrid ceiling collapse.

**Go/No-Go:**
- stable-only가 mixed slices에서 near-floor면 scaffold too weak → scaffold 확대
- stable-only가 mixed slices에서 near-ceiling면 scaffold too strong → scaffold 축소
- Week 2 이후 scaffold policy 동결

### 9.6 Risk F: oracle gap이 너무 작다

**문제:** heuristic allocation이 개선해도 oracle과 거의 차이가 없거나, 반대로 oracle gap이 너무 큰데 heuristic이 못 좁히면 story가 약해진다.

**Go/No-Go:**
- oracle–stable gap이 거의 없으면 해당 synthetic slice는 버림
- oracle gap이 큰데 hybrid가 거의 못 따라가면 signal/allocator redesign 필요

### 9.7 Risk G: mixed slice preregistration failure

**문제:** preregistered mixed slices에서 hybrid가 예상보다 안 이길 수 있다.

**정책:** slice 정의는 고정한다.  
결과가 안 맞으면 slice를 바꾸지 않고, theory prediction과 어긋난 원인 분석을 결과로 남긴다.

---

## 10. 실행 계획

## 10.1 Week 1–2: calibration sprint

가장 중요한 2주다. 여기서 논문의 절반이 결정된다.

### 할 일

1. stable scaffold v1 구현
2. cheap preselector 구현
3. scaffold-prefixed chunk loss 구현
4. cache span proposal 구현
5. Gemma 4 E2B bring-up + tokenizer/chat-template lock
6. Gemma 4 E4B bring-up + non-thinking/thinking decode lock
7. write adapter update 구현
8. end-to-end latency accounting logger 구현
9. RULER mini subset smoke test

### Week 2 종료 시 잠글 것

- default Gemma 4 variant
- Gemma 4 chat template / response parsing rule
- default scaffold policy
- default shortlist size K
- default chunk size
- default span length / cache unit
- budget grid
- mixed slice definitions
- subset ids / seeds / logging format

## 10.2 Week 3–6: synthetic closure

- RULER main sweep
- oracle upper bound
- K sensitivity
- route overhead study
- write–cache tradeoff curves

이 단계에서 core claim이 안 보이면, LongBench v2로 가지 않는다.

## 10.3 Week 7–12: realistic evaluation

- LongBench v2 main categories
- raw-length bucket reporting
- Budget-AUC
- mixed-slice analysis

## 10.4 Week 13–16: support and appendix

- ZeroSCROLLS support
- stronger comparator appendix
- 2D Pareto
- ablation matrix

## 10.5 Week 8부터 병렬 집필

실험 끝나고 쓰는 방식은 금지한다.

Week 8부터 항상 유지할 문서:

- Intro skeleton
- Related work
- Formulation
- Theory
- Methods figure draft

실험이 끝난 뒤에는 Results/Analysis만 채워 넣는 구조로 간다.

---

## 11. ICLR main text 설계 (9 pages)

### 권장 지면 배분

- Intro: 0.9p
- Related Work: 0.6p
- Problem Formulation: 0.9p
- Method: 1.4p
- Theory: 0.5p
- Experimental Setup: 0.8p
- Main Results: 2.2p
- Analysis/Ablations: 1.1p
- Limitations/Reproducibility/Ethics: 0.6p

### 메인 figures

1. **Figure 1:** method overview (stable scaffold + write + cache + allocator)
2. **Figure 2:** Pareto frontier on RULER + LongBench v2
3. **Figure 3:** failure-mode breakdown + K sensitivity inset

### 메인 tables

1. **Table 1:** main Budget-AUC + key budget points
2. **Table 2:** compact cost accounting (if space allows; otherwise appendix)

### appendix first pages

- 2D Pareto
- oracle gap table
- full ablation matrix
- exact subset policy
- implementation details

---

## 12. 집필 톤 규칙

최신 리뷰가 지적했듯, 지금 이 계획서의 가장 큰 강점 중 하나는 **claim의 절제**다. 실제 논문도 그 톤을 유지해야 한다.

### 절대 피할 표현

- “existing methods fail”
- “our method consistently outperforms all baselines”
- “this solves long-context reasoning”
- “novel architecture”
- “not RAG”

### 사용할 표현

- “existing methods often address one failure mode at a time”
- “we study the split between selective writing and selective exact caching”
- “the hybrid frontier dominates matched single-store specializations on mixed-failure slices”
- “the main finding is that the split matters; the specific allocator is one instantiation”

### abstract wording rule

`consistently improves` 같은 표현은 쓰지 않는다.  
대신 아래처럼 쓴다.

> improves the Pareto frontier on mixed-failure slices and improves aggregate Budget-AUC over matched single-store baselines

이 표현은 exact-only or diffuse-only slices에서 single-store가 강할 수 있다는 theory와도 충돌하지 않는다.

---

## 13. rebuttal playbook

### objection 1: “이건 그냥 RAG-like pipeline 아닌가?”

**답변:** 맞다, 구조적으로 bounded evidence materialization이 포함되므로 RAG-like 요소가 있다. 그러나 contribution은 retrieval 그 자체가 아니라, fixed extra latency budget 아래 exact retention과 parametric adaptation의 split을 정식화하고, equal-budget Pareto와 oracle gap으로 그 split의 중요성을 보인 데 있다.

### objection 2: “compute를 더 쓴 것뿐 아닌가?”

**답변:** 메인 budget axis는 measured extra end-to-end latency이며, route/signal/write/cache/decode overhead를 모두 포함한다. 또한 stable/write/cache/hybrid는 동일 frozen LM과 동일 scaffold를 공유한다.

### objection 3: “write branch가 실제로 뭘 하는지 불명확하다.”

**답변:** dependency-heavy synthetic slice에서 write-only의 효과를 먼저 검증했고, write–decode mismatch를 calibration 단계에서 체크했다. 또한 write objective와 cache objective는 명확히 분리되어 있으며, hybrid 이득은 mixed slices에서만 주장한다.

### objection 4: “K나 scaffold를 튜닝해서 이긴 것 아닌가?”

**답변:** K, scaffold rule, subset ids, mixed slices는 Week 2 이후 고정했고, main text에 K sensitivity와 scaffold calibration 결과를 넣었다.

### objection 5: “external SOTA와 왜 메인에서 직접 비교하지 않았나?”

**답변:** qTTT/GDWM/SR-TTT/WG-KV/Titans/ATLAS는 서로 다른 substrate와 regime를 전제한다. unfair head-to-head 대신, main claim을 동일 regime의 internal baselines로 제한하고, external baselines는 appendix에서 faithful setting이 가능할 때만 제공했다. 이는 claim과 evidence의 alignment를 우선한 선택이다.

---

## 14. 구현 체크리스트

### 반드시 로그로 남길 것

- wall-clock latency
- GPU-seconds
- route overhead
- write steps per chunk
- cached tokens/spans
- final materialized prompt length
- raw document length
- scaffold token count
- K
- chunk size
- seed
- decode config
- thinking on/off flag
- Gemma 4 variant

### 실험 freeze 전에 반드시 고정할 것

- Gemma 4 variant
- Gemma 4 chat template / response parsing rule
- scaffold construction rule
- K
- chunk size
- cache span length
- budget grid
- mixed slice list
- subset ids
- random seeds

### 금지 사항

- benchmark마다 scaffold를 다르게 튜닝하기
- 결과를 보고 mixed slice 다시 정의하기
- budget grid를 결과를 보고 바꾸기
- main text에서 external baselines에 과도하게 기대기

---

## 15. 최종 판정

이 계획서는 구현 가능한 연구 사양서 수준까지 정리되었다.

남아 있는 위험은 연구 방향의 위험이 아니라 구현상 subtle risk다. 그리고 그 subtle risk는 이미 다음 네 가지로 분해되었다.

1. scaffold-prefixed signal quality
2. write–decode distribution mismatch
3. route overhead vs budget ratio
4. preregistered mixed slices의 실제 behavior

이 네 가지는 모두 **Week 1–2 calibration**에서 답이 나온다.  
다음 단계는 캘리브레이션 착수다.

따라서 최종 실행 명령은 간단하다.

> **Week 1 immediately starts.**  
> Scaffold, K, budget accounting, and RULER smoke tests are the first milestone.  
> If calibration passes, the rest of the paper is an execution problem, not a research-definition problem.

---

## References / primary sources to anchor the paper plan

1. ICLR 2026 Author Guide: https://iclr.cc/Conferences/2026/AuthorGuide  
2. ICLR 2026 Reviewer Guide: https://iclr.cc/Conferences/2026/ReviewerGuide  
3. qTTT / “Let’s (not) just put things in Context”: https://openreview.net/forum?id=H0bcEdPCoc  
4. Gemma 4 launch blog: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/  
5. Gemma get started docs: https://ai.google.dev/gemma/docs/get_started  
6. Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4  
7. Gemma Hugging Face Transformers inference docs: https://ai.google.dev/gemma/docs/core/huggingface_inference  
8. Gemma LoRA tuning docs: https://ai.google.dev/gemma/docs/core/lora_tuning  
9. RULER paper/repo: https://github.com/NVIDIA/RULER  
10. LongBench v2 ACL 2025: https://aclanthology.org/2025.acl-long.183/  
11. ZeroSCROLLS: https://aclanthology.org/2023.findings-emnlp.536/  
12. HELMET repo / paper: https://github.com/princeton-nlp/HELMET  
13. SCBench: https://www.microsoft.com/en-us/research/publication/scbench-a-kv-cache-centric-analysis-of-long-context-methods/  
14. RAG (NeurIPS 2020): https://papers.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf  
15. Titans: https://arxiv.org/abs/2501.00663  
16. ATLAS: https://arxiv.org/abs/2505.23735  
17. GradMem: https://openreview.net/forum?id=Wdzhnmu5HR  
18. WG-KV: https://arxiv.org/abs/2512.17452
