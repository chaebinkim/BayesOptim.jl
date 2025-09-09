# BayesOptim

[![Build Status](https://github.com/sakibmatin/BayesOptim.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sakibmatin/BayesOptim.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Bayesian Optimization (Julia + Python) — 사용 가이드

이 저장소는 **Julia**에서 **PyCall**로 파이썬 BO 유틸을 불러, χ²(카이제곱) 최소화를 **가우시안 프로세스 기반 Bayesian Optimization(GP-BO)** 으로 수행합니다.
핵심 아이디어는 **목적함수**를

$$
y \;=\; -\log(\chi^2+\varepsilon)
$$

로 변환하여 **y를 최대화 ⇔ χ²를 최소화**로 바꾸는 것입니다. 수렴 후에는 GP 포스터리어를 이용해 **전역/국소 최소 가능성**을 시각화한 **2D 히트맵**도 자동 저장합니다.

---

## 설치 & 요구 사항

* Julia: `PyCall.jl`
* Python: `scikit-learn`, `numpy`, `scipy`, `pandas`, `matplotlib`
* 프로젝트 구조(예):

  ```
  src/
    Fit.jl
    Bopt.py
  ```

---

## 빠른 시작

```julia
include("src/Fit.jl")

# 목적함수: params::Dict 를 받아 χ²(양수)를 반환해야 함
Objective(params) = your_chi2_for(params)

# 탐색 구간 (각 파라미터 하한/상한)
interval = Dict(
  "J3"   => (0.0, 0.05),
  "J4"   => (-0.10, 0.00),
  "Jnnn" => (-0.05, 0.00),
  # ...
)

# 최적화 실행 (평가 횟수 예: 30*n)
Fit(Objective, interval, 180; file_name="Bopt_Log", fig_name="chi2")
```

**출력 파일**

* `Bopt_Log.csv` : 매 반복 로그 (아래 참조)
* `chi2_vs_Idx.png` : 반복 vs χ² 산포도/최소점
* `chi2_vs_params.png` : 각 파라미터 vs χ² 산포도/최소점
* `chi2_pair_<P>_vs_<Q>_Echi2.png` : (최종) 2D 기대 χ² 히트맵
* `chi2_pair_<P>_vs_<Q>_PI.png`     : (최종) 2D 개선확률(PI) 히트맵
* `Bopt_Log_uncert.json` : (최종) 포스터리어 샘플링 기반 최적값 불확실성 요약

**CSV 컬럼**

```
ID | <param1> ... <paramD> | Chi2 | Y
```

* `Y = -log(Chi2+eps)` 로 내부에서 변환됨(최대화 대상)

---

## Fit.jl — 함수 설명

### `Fit(Objective, interval, max_iter; file_name="Bopt_Log", fig_name="chi2")`

* **인자**

  * `Objective(::Dict) -> Real`: 파라미터 셋을 받아 **χ²(양수)** 를 반환.
  * `interval::Dict{String,(Float64,Float64)}`: 각 파라미터의 (하한, 상한).
  * `max_iter::Int`: 총 평가 예산(일반 권장치: 20·n \~ 40·n).

* **키워드 인자**

  * `file_name`: 로그/요약 파일 접두사.
  * `fig_name`: 그림 파일 접두사.

* **내부 동작 파이프라인**

  1. **PyCall로 Bopt.py 로드** → GP/BO 유틸 사용.
  2. **χ² → y 변환**: `sanitize_chi2`(비정상/음수/NaN 보호) 후 `y = -log(χ²+eps)`.
  3. **입력 정규화**: `UnitSpaceGP`로 모든 입력을 **\[0,1]^d** 에 매핑하여 스케일 이슈 해결.
  4. **초기 디자인**: `n_init = min(10, 2d)` 점 랜덤 평가.
  5. **GP 모델**: `Constant * Matern(ν=2.5, ARD) + WhiteKernel`, `normalize_y=True`.
  6. **신뢰영역(TuRBO-lite)**: 베스트 근처 박스의 한 변 길이 `L`을 성공/실패에 따라 확대/축소.
  7. **획득 & 제안**

     * 기본: **EI(Expected Improvement)** (최대화) + 상위 시작점 L-BFGS-B 정제.
     * **적응형 탐색률 ξ**: 정체가 길어질수록 ξ ↑ → 탐색 성향 강화.
     * **최소거리 제약**: 기존 점들과 유닛공간 거리 ≥ `min_dist` 인 후보를 우선 채택(군집/정체 방지).
     * **강제 탐색 스텝**: 정체/수축 시 한 번은 전역에서

       * Thompson 샘플링(`Propose_Thompson`) **또는**
       * 표준편차 최대(`Propose_MaxStd`)로 과감히 점프.
     * (옵션) **전역 PI 가드**: 전역에서 “개선 확률(PI)”이 충분히 크면 강제 탐색 유지.
  8. **로그/그림 저장**: CSV, 반복/파라미터 산포도 저장.
  9. **최종 단계(한 번만)**

     * **포스터리어 샘플링**으로 최적값/위치의 표준편차 추정 → `_uncert.json`.
     * **2D 히트맵** 자동 저장:

       * `mode="Echi2"`: **예측 기대 χ²** 지형(부드럽고 직관적)
       * `mode="PI"`: 현 베스트 대비 **개선 확률** 지도(남은 유망 영역 가시화)

* **튜닝 지표(코드 상단에서 수정 가능)**

  * 초기/최소 탐색률 및 감쇠: `xi0=0.05`, `xi_min=0.005`, `decay=0.5`
  * 정체 판정: 창 `W=10`, 상대개선 임계 `1e-3` (0.1%)
  * 신뢰영역: `L` 시작 0.8, `L_min=0.05`, `succ_th=3`, `fail_th=3`
  * 강제 탐색/PI 임계: `min_dist≈0.10~0.15`, `pi_far≈0.10`
  * 후보 수: EI `n_cand≈4096`, 전역 탐색/PI `3000~4000`

---

## Bopt.py — 함수/클래스 설명

### 경계/정규화

* ` _bounds_arrays(bounds, param_order) -> (lo, hi)`
  각 파라미터의 하한/상한 벡터 반환.
* ` to_unit_batch(bounds, param_order, X)`
  원래 스케일의 X를 **\[0,1]^d** 로 선형 변환.
* ` from_unit_batch(bounds, param_order, U)`
  유닛 공간의 U를 원래 스케일로 역변환.

### GP 래퍼

* `class UnitSpaceGP(model, bounds, param_order)`
  `fit/predict/sample_y`가 **자동으로 유닛 공간**에서 동작하도록 감싸는 래퍼.

  * `fit(X,y)`, `predict(X, return_std=True)`, `sample_y(X, n_samples)`

### 유틸

* `_as_2d(X)` : 1D를 (1,d)로 정규화.
* `_dedup_unit(x_u, X_u; tol)` : 유닛공간에서 거의 같은 점 존재 여부.
* `_sample_candidates(bounds, param_order; n_cand, trust_region, rng)`
  전역 또는 신뢰영역(유닛 공간 박스)에서 일괄 샘플링.

### GP 보조

* `surrogate(model, X) -> (mu, std)` : GP 예측 평균/표준편차.
* `Expected_Improvement(X_obs, XS, model, explore, y_obs)`
  관측 `y_obs`의 최대값을 기준으로 EI 계산(ξ=`explore`).

### 탐색(로컬 미니멈 회피)

* `_far_mask(bounds, param_order, XR, X; min_dist)`
  기존 점들과 유닛공간 거리 ≥ `min_dist` 인 후보 마스크.
* `Propose_Thompson(model, bounds, param_order; X, n_cand, min_dist)`
  **Thompson 샘플**에서 전역 최대. 멀리 떨어진 후보를 우선.
* `Propose_MaxStd(model, bounds, param_order; X, n_cand, min_dist)`
  표준편차(불확실성) 최대 후보. 멀리 떨어진 후보를 우선.
* `Global_PI(model, bounds, param_order, X, y; delta, n_cand, min_dist)`
  전역 후보에서 **개선 확률 PI**의 최대값(“새로운 지역”만 고려).

### 획득 최적화

* `Opt_Acquisition(X, y_obs, model, bounds; explore, n_cand, k_refine, param_order, trust_region, rng, min_dist)`
  EI 기반 제안:

  1. 후보 샘플→EI 상위 `k_refine`개로 시작
  2. 각 시작점에서 L-BFGS-B 정제
  3. **최소거리 제약**으로 군집 방지
     → (1,d) 제안점 반환

### 포스터리어 샘플링 (최종 1회)

* `optimal_std_via_sampling(model, bounds, param_order; X, y, n_funcs=200, n_cand=2000, trust_region, eps)`

  * 전역/신뢰영역에서 후보 `n_cand`개 샘플
  * GP로 **함수 전개(Thompson) `n_funcs`개**를 한꺼번에 샘플

    * 각 “함수 시나리오”마다 후보 중 최대값/위치 계산
    * `y*`와 역변환된 `χ²*`의 평균/표준편차, 위치 통계 반환
  * **`n_cand`** = 지도 해상도, **`n_funcs`** = 불확실성 추정의 몬테카를로 표본 수

### 2D 히트맵

* `_grid_for_pair(bounds, param_order, (p,q), x_fixed; grid_n)`
  `(p,q)` 2차원 격자 위에 **다른 파라미터는 x\_fixed**로 고정한 전 공간 점 생성.
* `pairwise_heatmap_plot(model, bounds, param_order, x_fixed; pairs, grid_n, mode, eta, X_hist, chi2_hist, out_prefix, eps)`

  * `mode="Echi2"`: $\mathbb{E}[\chi^2]=\exp(-\mu+0.5\sigma^2)-\varepsilon$ 맵
  * `mode="PI"`: 현 베스트보다 $(1+\eta)$배 더 작아질 **확률** 맵
  * 과거 평가점과 현재 베스트(`x_fixed`)를 오버레이, PNG 저장

### 재시작

* `Restart(bounds, file_name; param_order)`
  `file_name.csv`가 있으면 불러와 `X, y, idx_list` 반환(레거시 `Obj` 지원).

---

## 용어/하이퍼파라미터 요약

* `n_cand` : 후보 샘플 수(지도 해상도). 크면 전역 탐색/정제 품질↑, 비용↑.
* `k_refine` : EI 상위 시작점 개수(L-BFGS-B 다중 시작).
* `explore (ξ)` : EI의 탐색률. **적응형 스케줄**로 자동 조정.
* `L, L_min` : 신뢰영역 한 변 길이 / 최소 길이.
* `succ_th, fail_th` : 성공/실패 카운트 임계(확대/축소 트리거).
* `plateau, W` : 최근 `W`회 동안 상대개선이 작으면 plateau++.
* `min_dist` : 유닛공간 최소거리(새 지역 우선).
* `pi_far` : 전역 PI 임계(개선 여지 판단).
* `n_funcs` : 포스터리어에서 샘플 함수 개수(불확실성 추정의 몬테카를로 표본 수).

---

## 로컬 최적(국소)에 갇힘을 줄이는 장치

* **적응형 ξ** : 정체가 길면 ξ↑ → EI가 외부로 더 나감
* **최소거리 제약** : 같은 동네 반복 측정을 억제
* **강제 탐색(Thompson/Max-σ)** : 정체/수축 시 전역 점프 1회
* **전역 PI 가드** : “아직 좋아질 구역이 남았는가?”를 수치로 확인

---

## 자주 묻는 질문

* **χ²가 0 또는 NaN이면?**
  내부에서 `sanitize_chi2`가 큰 패널티 값(기본 `1e30`)으로 대체하여 수치 안정성 확보.
* **히트맵은 무엇을 의미?**

  * `Echi2`: GP가 보는 **평균적 지형**. 진짜 전역 최소처럼 보이는지 시각화.
  * `PI`: 현 베스트 대비 **개선 확률**. 남은 유망 영역이 있으면 붉게/푸르게 드러남.
