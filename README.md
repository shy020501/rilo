# RILO

**Robotic Interaction and Learning Orchestrator** — GPT가 관측을 보고 수치 action을 직접 생성하는 로봇 실험 프로토타입.

```text
Web UI ↔ Runner ↔ Policy (OpenAI Responses API)
             ↕
       EnvironmentAdapter → LIBERO / 다른 simulator / 실제 로봇
             ↓
       관측·action·실행 결과 기록
```

## 실행

Linux / Python **3.10**, 이름이 `rilo`인 Conda 환경을 사용한다. 로봇 시연 데이터 다운로드는 필요 없다.

```bash
git clone https://github.com/shy020501/rilo.git
cd rilo
conda create -n rilo python=3.10 pip
conda activate rilo
python -m pip install --upgrade pip
# CPU용 설치 예시. 이미 적절한 PyTorch가 있다면 생략 가능.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[libero,dev]'
python scripts/setup_libero.py

cp .env.example .env
# .env에 OPENAI_API_KEY 설정 또는 셸에서 export
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-6-astra"
export LIBERO_SUITE="libero_90"
export RILO_TASK_ID=7
rilo
```

**http://127.0.0.1:8000** → 프로젝트 선택 또는 생성 → 수집 화면에서 환경 초기화 → 수집 시작.
키 없이도 환경 초기화와 카메라 확인은 가능하다. 키는 서버에서만 읽으며, 변경하면 서버를 재시작한다.

- 메인 화면은 프로젝트별 demo 수, reset 결과, API 비용을 보여준다. 프로젝트 안에서 파이프라인 개요·수집 제어·episode 데이터·모델·설정을 확인한다.
- 프로젝트 생성과 화면 이동은 로봇 실행이나 API 호출을 시작하지 않는다. 초기화와 수집 시작은 별도 조작이다. 실행 중에는 **Pause**로 멈출 수 있다.
- 기존 API 실험은 환경·suite·task별 프로젝트에 자동 연결한다. 원본 run, 이미지, 비용 원장과 영상은 이동하거나 변경하지 않는다. 프로젝트 이름·목표·run 연결만 별도 `artifacts/projects.json`에 저장한다.
- **모델** 탭에서 DP 학습·평가, progress 학습, residual TD3와 checkpoint를 관리한다. **전체 자동 실행 / 단계별 실행**을 선택할 수 있다. 화면을 열거나 설정을 저장하는 것만으로 학습을 시작하지 않는다.
- 기존 상세 콘솔은 **/console**, 저장된 episode 영상은 프로젝트의 데이터 화면에서 확인한다.
- 프로젝트 **설정 → GPT 모델 설정**에서 모델과 추론 수준을 저장한다. 다음 **환경 준비** 또는 **복구된 환경에서 이어 수집**부터 적용되며, 이미 준비·실행·일시정지한 run의 설정은 유지된다. 모델 변경을 위해 서버를 재시작할 필요는 없다.
- 데이터의 각 episode에서 **보관**하면 기본 목록·demo 집계에서 빠지고 **보관함**에서 확인·복원할 수 있다. **삭제**는 확인 후 해당 episode의 관측·action 기록과 연결 영상을 영구 삭제한다. 이미 발생한 API 사용량·비용은 계속 누적 집계하며, 실행 중인 run의 데이터는 보관·삭제할 수 없다.
- 수집 화면의 실행 이벤트에는 episode 시작·결과·reset·Pause 같은 진행 상황을 표시한다. GPT가 반환한 동작 설명, action과 실행 결과는 **데이터 → episode 상세**에서 확인한다. 내부 추론 원문을 저장하거나 표시하는 기능은 아니다.

- **Pause / Resume**: 현재 한 control tick이 끝나면 멈춘다. 초기화/reset 중이면 해당 작업이 끝난 뒤 멈춘다. 대기 중인 GPT 요청과 미실행 action chunk를 폐기하고, 재개 시 새 관측으로 판단한다.
- **One decision**: GPT 판단 1회와 그 action chunk만 실행한 뒤 멈춘다.
- **Stop**: 현재 episode 종료. **Reset**은 기존 실행을 종료하고 같은 설정으로 새 run을 만든다.
- 기본 `GPT arm reset`은 성공 후에도 같은 물리 상태에서 GPT가 팔을 조작해 복구한다. **마지막 episode에도 reset을 수행한다.** Reset 실패·예산 소진 시 정지하며 자동으로 환경 reset하지 않는다.
- 기본 task는 `libero_90`의 **7번, 상단 서랍 열기**다. 주변 간섭이 적은 배치를 사용한다. `libero_goal` 0번 middle drawer와 `libero_90` 11번 top drawer도 독립 reset verifier를 지원한다. Reset은 서랍 닫기·gripper 열기·초기 팔 pose 근처 복귀·주변 물체 보존을 검사하며 초기 상태와 bit 단위로 동일하지는 않다.
- `episodes > 1`이면 검증된 GPT reset 후 다음 demo를 수집한다. `Environment reset`은 명시적으로 선택하는 별도 모드다. **Hard reset environment**는 현재 장면을 폐기하는 수동 복구 기능이다.
- 카메라, 실제 관측 시각, task, step/판단 수, action·실행 receipt·이벤트를 UI에서 확인할 수 있다.

키·시뮬레이터 없이 UI/제어 흐름만 확인하려면 `rilo --demo`를 사용한다. **명시적인 2D 테스트 환경과 scripted policy이며 LIBERO/GPT를 사용하지 않는다.**

## 설정

`.env.example` 참고. 주요 변수는 `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT`, `RILO_ENV`, `RILO_POLICY`, `LIBERO_SUITE`다. `OPENAI_BASE_URL`은 Responses·vision·function calling을 지원하는 호환 endpoint에만 사용한다.

`.env`의 모델·추론 수준은 프로젝트에서 별도로 지정하지 않았을 때 사용하는 서버 기본값이다. UI에서는 지원 모델별 추론 수준만 선택할 수 있고, 실행 manifest에 실제 적용한 모델·추론 수준을 기록한다. API key는 기존처럼 서버 환경 변수에서만 읽는다.

`RILO_PROMPT_FILE=prompts/control.md`를 설정하면 다음 GPT 판단부터 파일 수정을 읽으므로 물리 환경을 유지하며 프롬프트를 개선할 수 있다. 호출마다 정확한 프롬프트 hash와 snapshot을 기록한다. 실험 성공률 비교 시에는 프롬프트와 reasoning effort를 고정한다.

LIBERO는 `robosuite==1.4.1`, `mujoco==3.2.7`과 고정 upstream commit을 사용한다. 설치 스크립트가 `vendor/LIBERO`와 `artifacts/libero-config`를 만들며 기존 `~/.libero`를 변경하지 않는다.

렌더링은 기본 `MUJOCO_GL=osmesa`(Linux의 `libOSMesa` 필요). NVIDIA 드라이버가 정상인 머신은 `MUJOCO_GL=egl`을 사용할 수 있다. 첫 초기화에는 shader/JIT 준비 시간이 든다. `rilo doctor`로 설치 상태를 확인한다.

서버는 기본 localhost에만 열린다. 원격 확인은 `ssh -L 8000:127.0.0.1:8000 user@robot-host`로 터널링한다. 이 프로토타입에는 사용자 인증이 없으므로 공개 인터넷에 노출하지 않는다. **프로세스/worker는 하나만 실행한다.**

## 구조와 확장

| 위치 | 책임 |
|---|---|
| `src/rilo/contracts.py` | `Observation`, `ActionSpec`, `Decision`, adapter/policy 계약 |
| `src/rilo/adapters/libero.py` | LIBERO 관측·초기화·action 해석·성공 predicate |
| `src/rilo/policies/openai_policy.py` | 이미지·상태·최근 receipt → 검증된 수치 action |
| `src/rilo/runner.py` | 제어권, Pause/Resume, 실행 예산, episode 전환 |
| `src/rilo/recording.py` | control tick별 관측과 실행 기록 |
| `src/rilo/server.py`, `src/rilo/web/` | 환경에 독립적인 HTTP API와 웹 콘솔 |
| `src/rilo/learning/` | dataset, Diffusion Policy, progress, residual TD3·replay |
| `src/rilo/pipeline.py` | 순차 수집·학습·평가, 실행 권한과 모델 레지스트리 |

새 환경은 `action_spec`, `describe`, `reset`, `observe`, `step`, `pause`, `close`를 구현하고 `RILO_ENV=my_package.adapter:create_adapter`로 지정한다. `list_tasks`는 선택 사항이다. Factory는 인자 없이 adapter를 반환한다. Action 차원·단위·좌표계·gripper 의미는 **adapter가 선언**하며 runner/policy에 LIBERO의 7차원 가정이 없다. 새 정책도 `decide`와 `close`를 구현해 `RILO_POLICY=module:factory`로 연결할 수 있다.

GPT reset을 지원하는 환경은 `begin_reset`과 `begin_task`도 구현한다. 두 메서드는 물리 상태를 유지하며 목표만 바꾸고, `Observation.success`는 현재 task/reset 목표의 독립 검증 결과다. `/api/continue`는 검증된 GPT reset 상태에서 환경 초기화 없이 새 run을 시작한다.

모든 simulator 접근은 같은 전용 thread에서 이루어진다. `step`은 한 번의 제한된 control tick, `observe`는 동작 없는 새 관측이다. 실제 로봇 adapter는 `pause`를 로컬 정지에 연결하고 controller watchdog을 별도로 구현해야 한다. 웹 Pause는 물리 E-stop을 대체하지 않는다.

LIBERO action은 **정규화된 OSC delta**이며 직접적인 metre 명령이 아니다. Translation/rotation scale과 repeat 의미를 adapter가 GPT에 전달한다. Object ground-truth pose는 GPT 입력에 포함하지 않고, 성공은 LIBERO predicate로 확인한다. GPT의 성공 주장만으로 성공 데이터를 만들지 않는다.

한 GPT 응답은 최대 8개 action vector를 담으며, 각 vector는 최대 10 tick 반복하고 전체 chunk는 최대 30 tick이다. 현재 LIBERO는 20 Hz이므로 tick 하나는 시뮬레이션 0.05초다. GPT 응답 대기 중에는 `env.step()`을 호출하지 않아 시뮬레이션 시간이 흐르지 않는다. 실제 로봇의 대기 중 자세 유지와 상태 변화 처리는 별도 로컬 controller의 책임이다.

## 기록과 검증

`artifacts/runs/<run_id>/`에 실행 설정, 환경 설명, `episode-*/transitions.jsonl`, 카메라 JPEG, episode summary를 저장한다. 요청 action과 실제 적용 action, 관측 ID·시간, 부분 실행 receipt를 남긴다. Reset 구간은 task transition과 분리한다. History는 최근 실행 receipt로 제한하며, 모델에는 현재 카메라 이미지와 측정 상태를 함께 보낸다.

**실제 실행 데이터는 Git에 올리지 않는다.** `.env`와 그 변형, `artifacts/` 전체(run·이미지·영상·비용 원장·프로젝트 상태·학습 checkpoint), 가상환경과 `vendor/`는 `.gitignore`로 제외한다. 저장소에는 코드·테스트·설정 예시만 포함하며, 기존 로컬 데이터는 그대로 유지된다.

```bash
python -m pytest -q
```

테스트는 API 호출 없이 action 검증, 잘못된 성공 주장, 제어 예산, 기록, Pause 중단·늦은 응답 폐기·재개, GPT reset 순환, 비용 계산, 웹 API를 확인한다.

각 run의 `api_calls.jsonl`에는 유효하지 않은 응답까지 포함한 API 호출별 raw usage와 추정 비용을 기록한다. Input/cache read/cache write/output을 구분하고 reasoning token은 output의 부분집합으로 표시한다. 응답을 받지 못한 timeout·취소 등은 비용 미확인으로 남긴다. UI 달러 예산은 응답 수신 뒤 확인하므로 마지막 한 요청 비용만큼 초과할 수 있다. 비용은 [공식 단가](https://developers.openai.com/api/docs/models/gpt-6-astra) 기준 추정이며 실제 청구서와 다를 수 있다.

`python scripts/report_costs.py artifacts/runs/<run_id> --json artifacts/costs.json --markdown artifacts/costs.md`로 task/reset 평균과 실패 비용까지 포함한 성공 demo당 비용을 다시 계산한다. 여러 run 경로를 전달하면 모델·reasoning effort·코드 버전별로 구분한다.

## 학습 파이프라인

`pip install -e '.[learning,dev]'` 후 프로젝트 **모델** 탭에서 설정한다.

**GPT 수집 → DP 학습 → BC 평가 → 목표 미달 시 반복 → progress 학습 → residual TD3 / 평가 반복**. 수집과 supervised 학습은 순차 실행한다. 성공·실패를 모두 기록하고, DP·progress는 기본적으로 성공 task 구간만, RL replay는 두 종류를 모두 사용한다. Reset은 GPT action으로 수행하며 자동 simulator reset은 하지 않는다.

DP는 ResNet18 기반 약 13.7M 모델, residual은 frozen DP 위의 작은 MLP다. 모델·지표·설정은 `artifacts/learning/<project_id>/`에 저장한다. **실제 데이터 학습과 성능 측정은 아직 실행하지 않았다.** 합성 fixture로 전체 6단계 smoke test를 제공한다.

[모델 구조·실행 방법·설계 차이](docs/learning.md) · [기본 hyperparameter](configs/pipeline.json). CLI는 `python scripts/run_pipeline.py --project <ID> status`로 상태를 확인하고 `start --stage bc`, `start --stage rl`, `start --auto`로 명시적으로 실행한다.

**현재 범위:** GPT 직접 제어, LIBERO, 데이터 관리, DP·progress·residual RL과 순차 orchestration/UI. 실제 로봇 adapter와 학습 성능 평가는 후속 검증이 필요하다.

## 실제 측정 · 2026-10-27

`gpt-6-astra / xhigh`, `libero_90 / task 7`에서 같은 초기 배치로 연속 3회 수행했다. Task와 GPT reset 모두 **3/3 성공**, 사이클 사이 환경 reset **0회**였다. 3회 소표본이며 다른 배치/seed로 일반화된 성공률은 아니다.

Demo 1개당 평균 **task 129,564 tokens / $1.62**, **reset 112,794 tokens / $1.28**, **합계 242,358 tokens / $2.90 / 약 7.8분**. 3회 합계 $8.70, 실패·디버깅·task 선택 실험까지 포함한 전체 추정 비용은 **$17.95**였다. 초기 middle-drawer 장면은 접시 복구에 실패하여 최종 task에서 제외했다.

측정 원본은 로컬의 `artifacts/final-costs.md`, `artifacts/all-experiments-costs.md`, `artifacts/collection-campaign.json`에 보관하며 GitHub에는 포함하지 않는다.

저장된 5개 episode 영상은 `artifacts/videos/index.html`에서 확인한다. 외부/손목 카메라와 TASK/RESET 표시가 포함되며 **API 대기 시간을 제외한 시뮬레이션 시간 기준 20 fps**다.

```bash
python scripts/serve_videos.py
# http://127.0.0.1:8002
```

다른 기록은 `python scripts/export_videos.py --help`의 옵션으로 MP4로 변환할 수 있다. 이 변환은 저장된 이미지만 사용하며 로봇 실행이나 API 호출을 하지 않는다.
