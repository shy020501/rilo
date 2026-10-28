# 데이터 수집·학습 파이프라인

**서버 시작, UI 접속, 설정 저장은 학습이나 GPT 호출을 시작하지 않는다.** 프로젝트의 **모델** 탭에서 전체 자동 실행 또는 단계별 실행을 명시적으로 누른다. 같은 서버의 CLI도 같은 실행 권한을 사용한다.

## 실행 순서

1. `collect`: 지정한 episode 수를 GPT로 수행하고 GPT가 팔을 움직여 reset한다. 성공과 실패를 모두 기록한다.
2. `bc`: 수집을 멈추고 성공한 **task** 구간으로 Diffusion Policy를 학습한다. 실패 기록은 보존하며 RL에서 사용한다.
3. `bc_eval`: DP만으로 평가하고 GPT로 reset한다. 성공·실패 모두 저장한다. 목표 성공률 미달 시 1–3을 반복한다.
4. `progress`: 현재 BC가 평가 목표를 달성하면 성공 trajectory만으로 progress 모델을 학습한다.
5. `rl`: BC 전체를 동결하고 매 control tick마다 residual action을 학습한다. 1–4에서 쌓인 성공·실패 데이터를 offline replay로 사용한다.
6. `rl_eval`: exploration과 parameter update 없이 BC+residual을 평가한다. 목표 미달 시 5–6을 반복한다.

`max_bc_cycles`, `max_rl_rounds`, run별 API 예산과 task/reset step 한도로 실행을 제한한다. GPT reset 검증 실패 시 중지하고 사용자 복구를 기다린다. **반복 사이 simulator reset을 자동으로 호출하지 않는다.** 최초 환경 준비 이후 검증된 GPT reset 장면을 유지한다. 실패·중단 뒤 검증된 reset이 없으면 수집 화면에서 명시적으로 복구해야 한다.

학습 작업이 실행 권한을 보유하는 동안 다른 수집 시작·환경 초기화 및 해당 프로젝트의 데이터 보관·삭제를 막는다. Pause는 학습 update 경계에서 대기하며 로봇의 후속 tick도 차단한다. Stop은 협조적 취소이므로 진행 중인 짧은 연산이 끝난 뒤 종료된다. 서버 재시작 시 미완료 작업은 `interrupted`로 표시하며 자동 재개하지 않는다.

## 모델

### BC: Diffusion Policy

- 공유 **ResNet-18 + GroupNorm + spatial softmax 32 keypoints**를 각 카메라에 적용하고 proprioception과 결합한다. 이미지 128×128, 관측 history 2개, feature 128차원이 기본값이다.
- FiLM 조건부 1D temporal U-Net 채널 64/128/256, 예측 horizon 16, 실행 chunk 4, 학습 diffusion 100 step, 추론 DDIM 10 step. LIBERO 2-camera/7-action 구성은 약 **13.7M parameters**다.
- ImageNet 초기화는 학습 시작 시에만 필요하면 다운로드한다. `tiny` encoder는 `imagenet_init=false`와 함께 합성 smoke test에 사용한다.
- 실제 `applied_action`을 학습한다. 이미지·state schema와 action bounds를 checkpoint에 저장하며 action 차원이나 카메라 이름에 로봇별 상수를 넣지 않는다.
- Episode 단위 train/validation split으로 누출을 막고, padding action은 loss에서 제외한다. 최소 2개의 성공 episode가 필요하다. validation diffusion loss로 EMA checkpoint와 early stopping을 선택하고, 성공률은 별도 rollout으로 측정한다.
- 기본 BC는 성공 데이터만 사용한다. `bc.successful_only=false`로 변경할 수 있지만 실패 동작까지 모방하게 된다는 점을 고려해야 한다.

### Progress

BC에서 학습한 동결 encoder + 128-hidden GRU가 최근 4개 관측 feature를 받아 0–1 progress를 예측한다. 성공 trajectory 내 정규화된 위치를 약한 정답으로 사용하고 회귀 loss와 temporal ranking loss를 함께 학습한다. 별도 성공 episode에서 validation MAE를 계산한다.

Progress는 성공 판정 자체가 아니다. RL task 수행 중 기본 0.8 이상이면 20 control step 간격으로 GPT 판정을 요청한다. GPT 판정은 이동 action을 생성할 수 없으며 관측 근거와 성공·실패·미확정 결과를 기록한다. **LIBERO 성공 라벨과 보상은 환경의 독립 predicate를 우선한다.** GPT 성공 주장만으로 실패 episode를 성공으로 바꾸지 않는다. 실제 로봇으로 확장할 때는 adapter의 성공 검증 계약도 별도로 구현해야 한다.

### Residual TD3

`a = clip(a_BC + residual_scale × half_action_range × tanh(actor(features, a_BC)))`

- 동결 DP의 chunk를 유지하면서 현재 관측을 매 tick 읽는다. `observe/peek`는 chunk를 소비하지 않고 실제 action 실행 뒤 `advance`만 한 번 호출한다.
- Actor는 LayerNorm MLP 256×256, 마지막 층 0 초기화. 초기 deterministic residual은 정확히 0이다. 기본 보정 범위는 각 action 범위의 절반의 0.1배이며, exploration·target smoothing에도 같은 제한을 적용한다.
- Critic은 **실제 실행한 full action**의 Q를 학습한다. 기본 Q ensemble 4개, target에는 무작위 2개의 최솟값, actor에는 ensemble 평균을 사용한다. `num_critics=2`로 twin critic을 선택할 수 있다.
- 기본 offline/online 50:50 sampling, n-step=3, UTD=4, delayed actor update=2, Polyak tau=0.005. LayerNorm과 target policy smoothing을 포함한다. Replay가 비어 있을 때의 fallback은 metrics에 실제 비율로 남긴다.
- 성공·실패 offline trajectory 모두 frozen DP로 **순차 재생하여 chunk에 맞는 BC action을 재구성**한다. 시연 action을 BC 예측값으로 간주하지 않는다. Offline action label은 실제 실행 action 그대로 유지한다.
- 성공 terminal reward=1, 나머지=0. N-step은 episode/task-reset 경계를 넘지 않는다. 시간 제한·외부 중단은 truncation이며 기본적으로 bootstrap한다. 실제 terminal은 bootstrap하지 않는다.
- DP encoder까지 동결하며 cached feature replay를 사용한다. 이는 별도 vision encoder까지 RL로 업데이트하는 원 ResFiT의 구성을 단순화한 선택이다. 현재 구현은 **ResFiT 스타일**이며 원 논문의 전체 실험 재현을 주장하지 않는다.
- Replay·optimizer·target networks·RNG를 residual checkpoint에 저장한다. 같은 BC fingerprint와 같은 residual 설정이면 이전 checkpoint를 재개한다. Residual 설정을 변경하면 새 residual 학습을 시작하고 이전 모델은 보존한다. 중간 중지 시에도 수집된 update를 partial checkpoint로 저장한다. BC-only와 residual 평가는 같은 episode 수·seed·GPT reset 프로토콜로 비교한다.

실제 성공률·소요 시간·GPU 사용량은 아직 측정하지 않았다. 같은 seed의 연속 GPT reset 평가이므로 서로 다른 초기 배치에 대한 일반화 성능과 구분해야 한다.

## 설정과 명령

```bash
conda activate rilo
python -m pip install -e '.[learning,dev]'
# GPU를 사용할 경우 장비에 맞는 CUDA PyTorch를 설치한다.
python -m rilo --port 8003

# 아래 status/configure는 학습을 시작하지 않는다.
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> status
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> configure --file configs/pipeline.json

# 실제 실행을 원할 때만 호출한다. GPT 수집/reset/판정은 API 비용이 발생한다.
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> start --auto
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> start --stage bc
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> start --stage bc_eval
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> start --stage progress
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> start --stage rl
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> start --stage rl_eval
python scripts/run_pipeline.py --server http://127.0.0.1:8003 --project <PROJECT_ID> pause
```

프로젝트 URL의 ID를 사용한다. 서버 프로세스는 하나만 띄운다. `configs/pipeline.json`과 모델 탭의 고급 JSON에서 batch size, learning rate, horizon, encoder, n-step, UTD, replay 비율, 보정 범위 등을 변경할 수 있다. `device=auto`는 CUDA 사용 가능 여부에 따라 선택한다.

학습 결과는 `artifacts/learning/<project_id>/registry.json` 및 `models/<model_id>/`에 저장한다. checkpoint, 학습 설정, 원본 episode·이미지 hash를 포함한 dataset provenance, validation metrics를 보존한다. 원본 데이터는 기존 `artifacts/runs/`에 남으며 snapshot은 참조와 hash를 고정한다. 보관·삭제된 episode와 reset 동작은 task 학습에서 제외한다. 평가·online rollout은 기존 recorder에 기록하고 `manifest.config.learning`으로 사용한 모델과 단계를 구분한다.

## 검증

```bash
python -m pytest -q
```

테스트는 임시 synthetic trajectory와 작은 random encoder를 사용한다. 실제 데이터 학습, ImageNet 다운로드, 유료 GPT 호출은 하지 않는다. DP chunk 소비, action clipping, frozen weights, TD3 target·delayed update, replay 혼합·n-step 경계, checkpoint 재로딩, Pause/Stop, 전체 단계 전환을 확인한다.

## 참고한 원문

- [Diffusion Policy 프로젝트·논문·공식 코드](https://diffusion-policy.cs.columbia.edu/)
- [ResFiT 논문](https://arxiv.org/abs/2509.19301), [공식 코드](https://github.com/amazon-far/residual-offpolicy-rl)

원 논문의 per-step residual, full-action Q, demo 혼합, n-step, Q ensemble, LayerNorm, 높은 UTD 구성을 참고했고, RILO의 관측·action·reset 계약에 맞게 독립 구현했다.
