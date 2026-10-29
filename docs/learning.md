# 데이터 수집·학습 파이프라인

**서버 시작, UI 접속, 설정 저장은 학습이나 GPT 호출을 시작하지 않는다.** 전체 자동 실행은 **개요**, 개별 학습은 **모델**, 개별 평가는 **평가** 탭에서 시작한다. 같은 서버의 CLI도 같은 실행 권한을 사용한다.

## 실행 순서

1. `collect`: 지정한 episode 수를 GPT로 수행하고 GPT가 팔을 움직여 reset한다. 성공과 실패를 모두 기록한다.
2. `bc`: 수집을 멈추고 성공한 **task** 구간으로 선택한 BC 모델을 학습한다. 현재 구현은 Diffusion Policy이며, 실패 기록은 보존해 RL에서 사용한다.
3. `bc_eval`: BC 정책만으로 평가하고 실행 팝업에서 선택한 방식으로 reset한다. 성공·실패 모두 저장한다. 목표 성공률 미달 시 1–3을 반복한다.
4. `rl`: 현재 BC가 평가 목표를 달성하면 RL 학습을 실행한다. 조건에 맞는 Progress 모델을 재사용하고, 없거나 오래된 경우 성공 trajectory로 먼저 학습한다. 이후 BC 전체를 동결하고 매 control tick마다 residual action을 학습하며 기존 성공·실패 데이터를 offline replay로 사용한다.
5. `rl_eval`: exploration과 parameter update 없이 BC+residual을 평가한다. 목표 미달 시 4–5를 반복한다.

모델 탭은 **학습 진행 상황**(BC 학습·RL 학습 두 단계), BC 학습, BC 지표·기록, RL 학습, RL 지표·기록, 모델 레지스트리 순으로 배치한다. BC 모델 선택에서 DP를 실행할 수 있고, pi0·pi0.5는 미구현 항목으로 표시한다. RL은 현재 Residual TD3를 지원한다. Progress는 RL의 내부 준비 단계이며 별도 학습 버튼이나 단계 카드를 만들지 않는다. RL 실행 팝업의 **Progress 모델 사용**은 현재 필수로 체크·잠금 상태이며 해제할 수 없다. 기존 CLI의 `progress` 단계는 유지한다.

BC·RL의 학습 버튼을 누르면 해당 모델의 실행 설정과 하이퍼파라미터 팝업이 열린다. 평가 탭에서는 결과·기록을 먼저 보여주고, **선택 평가 실행**을 누르면 평가 설정 팝업을 연다. 팝업을 열거나 취소하면 설정 저장·학습·평가를 시작하지 않으며, **설정 저장 후 실행**을 눌러야 선택한 단계를 실행한다.

**BC 학습 지표·로그**의 **학습 step / Train loss / Validation loss** 카드를 누르면 해당 학습의 기록 팝업을 연다. 학습 ID를 고정한 채 2.5초마다 저장된 `metrics.jsonl`을 읽으며, 완료·중단된 학습도 기록이 있으면 조회할 수 있다. Train loss는 실제 저장된 지점을 연결하고 validation loss는 실제 평가 지점만 표시한다. 기본 train 로그 간격은 최대 25 steps, validation은 `eval_every` 및 시작·마지막 step이다. 큰 로그는 최근 8 MiB 내에서 일반 train 지점을 줄여 표시하며, 포함된 validation 지점은 유지한다. 제한·샘플링 여부는 팝업에 표시한다. 체크포인트의 실제 저장 step과 파일 목록도 함께 확인할 수 있다.

`max_bc_cycles`, `max_rl_rounds`, run별 API 예산과 task/reset step 한도로 실행을 제한한다. GPT reset 모드는 최초 환경 준비 이후 검증된 복구 장면을 유지하며, 복구 검증에 실패하면 멈춘다. 안전 후퇴 모드는 대상 복구와 후퇴를 검증한 뒤 시뮬레이터를 초기화한다.

**평가 → 선택 평가 실행 → 평가 Reset 방식**은 BC와 RL 평가에 공통 적용된다. `evaluation_reset_mode`는 `inherit`(수집 설정 따르기), `agent`(GPT reset), `environment`(환경 reset), `free_space`(복구·안전 후퇴 후 환경 reset) 중 선택한다. 개별 평가와 전체 파이프라인 평가가 같은 설정을 사용하며, 수집과 RL 학습의 reset 설정은 별도로 유지한다. 환경 reset 평가는 시뮬레이터에서 첫 episode를 설정된 seed로 초기화하고, 다음 episode마다 seed를 1씩 늘려 초기화한다. 마지막 episode 뒤에는 추가 reset을 하지 않는다. 이 모드의 BC 평가는 GPT reset·판정·grasp 기준 추정 호출 없이 실행된다. RL 평가는 환경 reset을 선택해도 기존 progress 기반 GPT 판정을 사용할 수 있다.

평가 탭의 실시간 카메라는 현재 프로젝트의 BC/RL 평가 rollout을 표시한다. Pause와 같은 run의 reset 준비 중에는 마지막 이미지를 유지하고, 다음 관측이 도착하면 갱신한다. 다른 프로젝트나 수집 run의 화면을 평가 화면으로 표시하지 않는다. 완료된 평가의 episode와 영상 생성 기능은 데이터 탭에서 확인한다. 평가 준비를 시작할 때 실행 한도·seed·GPT 설정을 고정하며, 준비 중 저장한 변경은 다음 실행에 적용한다. 평가 결과에 실제 reset 방식과 프로토콜을 기록하며, reset 방식이 바뀌면 이전 평가를 현재 조건의 완료 근거로 재사용하지 않는다.

학습 작업이 실행 권한을 보유하는 동안 다른 수집 시작·환경 초기화 및 해당 프로젝트의 데이터 보관·삭제를 막는다. Pause는 학습 update 경계에서 대기하며 로봇의 후속 tick도 차단한다. Stop은 협조적 취소이므로 진행 중인 짧은 연산이 끝난 뒤 종료된다. 서버 재시작 시 미완료 작업은 `interrupted`로 표시하며 자동 재개하지 않는다.

## 모델

### BC: Diffusion Policy

- 공유 **ResNet-18 + GroupNorm + spatial softmax 32 keypoints**를 각 카메라에 적용하고 proprioception과 결합한다. 이미지 128×128, 관측 history 2개, feature 128차원이 기본값이다.
- FiLM 조건부 1D temporal U-Net 채널 64/128/256, 예측 horizon 16, 실행 chunk 4, 학습 diffusion 100 step, 추론 DDIM 10 step. LIBERO 2-camera/7-action 구성은 약 **13.7M parameters**다.
- ImageNet 초기화는 학습 시작 시에만 필요하면 다운로드한다. `tiny` encoder는 `imagenet_init=false`와 함께 합성 smoke test에 사용한다.
- 실제 `applied_action`을 학습한다. 이미지·state schema와 action bounds를 checkpoint에 저장하며 action 차원이나 카메라 이름에 로봇별 상수를 넣지 않는다.
- Episode 단위 train/validation split으로 누출을 막고, padding action은 loss에서 제외한다. 최소 2개의 성공 episode가 필요하다. validation diffusion loss로 EMA checkpoint와 early stopping을 선택하고, 성공률은 별도 rollout으로 측정한다.
- BC 실행 팝업의 **체크포인트 저장 간격**(`bc.checkpoint_every`, 기본 1,000 steps)은 optimizer·EMA update가 끝난 뒤 중간 상태를 저장하는 간격이다. `checkpoint-step-00001000.pt` 같은 파일에 EMA 추론 가중치와 원본 모델·optimizer·Torch RNG·loader 상태를 함께 저장하며 `checkpoints.json`에 실제 저장 목록을 기록한다. 정상 종료나 early stopping 시 마지막 step이 주기 저장과 겹치지 않으면 종료 상태도 저장한다. `policy.pt`는 기존처럼 validation loss가 가장 좋은 EMA 정책으로 유지한다. 중간 파일은 완료 모델로 등록하지 않는다.
- Pause는 메모리 상태를 유지한 채 update 경계에서 대기하며, Pause 자체가 체크포인트 저장을 강제하지는 않는다. 중지·오류 시 이미 저장된 중간 파일은 남지만, **디스크 체크포인트에서 BC 학습을 재개하는 기능은 아직 없다.** 저장 간격 변경은 다음 학습부터 적용하고, 간격만 변경해도 기존 완료 BC·평가 결과를 무효화하지 않는다.
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
- Replay·optimizer·target networks·RNG를 residual checkpoint에 저장한다. 같은 BC fingerprint와 같은 residual 설정이면 이전 checkpoint를 재개한다. Residual 설정을 변경하면 새 residual 학습을 시작하고 이전 모델은 보존한다. 중간 중지 시에도 수집된 update를 partial checkpoint로 저장한다. BC-only와 residual 평가는 같은 episode 수·seed·reset 설정으로 비교한다.

실제 성공률·소요 시간·GPU 사용량은 아직 측정하지 않았다. 연속 GPT reset 평가와 episode별 환경 초기화 평가는 조건이 다르므로 결과에 기록된 reset 프로토콜을 함께 확인해야 한다.

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

프로젝트 URL의 ID를 사용한다. 서버 프로세스는 하나만 띄운다. `configs/pipeline.json`과 학습 실행 팝업의 고급 JSON에서 batch size, learning rate, horizon, encoder, n-step, UTD, replay 비율, 보정 범위 등을 변경할 수 있다. 기본 학습 장치는 `cuda:0`이다. CUDA를 사용할 수 없으면 CPU로 조용히 전환하지 않고 실행 오류를 표시한다. CPU 실행은 `device=cpu`, 자동 선택은 `device=auto`로 명시적으로 바꿀 수 있다.

학습 결과는 `artifacts/learning/<project_id>/registry.json` 및 `models/<model_id>/`에 저장한다. checkpoint, 학습 설정, 원본 episode·이미지 hash를 포함한 dataset provenance, validation metrics를 보존한다. 원본 데이터는 기존 `artifacts/runs/`에 남으며 snapshot은 참조와 hash를 고정한다. 보관·삭제된 episode와 reset 동작은 task 학습에서 제외한다. 평가·online rollout은 기존 recorder에 기록하고 `manifest.config.learning`으로 사용한 모델과 단계를 구분한다. 데이터 목록·상세·평가 화면에서 BC 평가, RL 학습 및 RL 평가 기록은 **Rollout 01** 형식으로 표시한다. 수집·외부 시연은 **Episode 01** 형식을 유지하며, 기존 프로젝트 표시 번호와 저장 경로는 변경하지 않는다.

## 검증

```bash
python -m pytest -q
```

테스트는 임시 synthetic trajectory와 작은 random encoder를 사용한다. 실제 데이터 학습, ImageNet 다운로드, 유료 GPT 호출은 하지 않는다. DP chunk 소비, action clipping, frozen weights, TD3 target·delayed update, replay 혼합·n-step 경계, checkpoint 재로딩, Pause/Stop, 전체 단계 전환을 확인한다.

## 참고한 원문

- [Diffusion Policy 프로젝트·논문·공식 코드](https://diffusion-policy.cs.columbia.edu/)
- [ResFiT 논문](https://arxiv.org/abs/2509.19301), [공식 코드](https://github.com/amazon-far/residual-offpolicy-rl)

원 논문의 per-step residual, full-action Q, demo 혼합, n-step, Q ensemble, LayerNorm, 높은 UTD 구성을 참고했고, RILO의 관측·action·reset 계약에 맞게 독립 구현했다.
